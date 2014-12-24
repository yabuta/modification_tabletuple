/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This file contains original code and/or modifications of original code.
 * Any modifications made by VoltDB Inc. are licensed under the following
 * terms and conditions:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */
/* Copyright (C) 2008 by H-Store Project
 * Brown University
 * Massachusetts Institute of Technology
 * Yale University
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <vector>
#include <string>
#include <stack>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nestloopexecutor.h"
#include "common/debuglog.h"
#include "common/common.h"
#include "common/tabletuple.h"
#include "common/FatalException.hpp"
#include "common/types.h"
#include "executors/aggregateexecutor.h"
#include "execution/ProgressMonitorProxy.h"
#include "expressions/abstractexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "expressions/comparisonexpression.h"
#include "storage/table.h"
#include "storage/temptable.h"
#include "storage/tableiterator.h"
#include "plannodes/nestloopnode.h"
#include "plannodes/limitnode.h"
#include "plannodes/aggregatenode.h"
#include "GPUNIJ.h"
#include "GPUSHJ.h"
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/common/GTupleSchema.h"
#include "GPUetc/expressions/makeexpressiontree.h"
#include "GPUetc/expressions/Gabstractexpression.h"
#include "GPUetc/expressions/nodedata.h"

#ifdef VOLT_DEBUG_ENABLED
#include <ctime>
#include <sys/times.h>
#include <unistd.h>
#endif


namespace voltdb{


class Test{
public:
  Test(){};

};
}


using namespace std;
using namespace voltdb;


bool NestLoopExecutor::p_init(AbstractPlanNode* abstract_node,
                              TempTableLimits* limits)
{
    VOLT_TRACE("init NestLoop Executor");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(abstract_node);
    assert(node);

    // Create output table based on output schema from the plan
    setTempOutputTable(limits);

    assert(m_tmpOutputTable);

    // NULL tuple for outer join
    if (node->getJoinType() == JOIN_TYPE_LEFT) {
        Table* inner_table = node->getInputTable(1);
        assert(inner_table);
        m_null_tuple.init(inner_table->schema());
    }

    // Inline aggregation can be serial, partial or hash
    m_aggExec = voltdb::getInlineAggregateExecutor(m_abstractNode);

    return true;
}


/**
copy m_data,SourceInlined and ValueType from NValue to GNValue
 */
void setGNValue(COLUMNDATA *cd,NValue NV){

  cd->gn.setMdata(NV.getMdataForGPU());
  cd->gn.setSourceInlined(NV.getSourceInlinedForGPU());
  cd->gn.setValueType(NV.getValueTypeForGPU());

}

void swap(RESULT *a,RESULT *b)
{
  RESULT temp;
  temp = *a;
  *a=*b;
  *b=temp;
}

void qsort(RESULT *jt,int p,int q)
{
  int i,j;
  int pivot;

  i = p;
  j = q;

  pivot = jt[(p+q)/2].lkey;

  while(1){
    while(jt[i].lkey < pivot) i++;
    while(pivot < jt[j].lkey) j--;
    if(i>=j) break;
    swap(&jt[i],&jt[j]);
    i++;
    j--;
  }
  if(p < i-1) qsort(jt,p,i-1);
  if(j+1 < q) qsort(jt,j+1,q);

}



bool NestLoopExecutor::p_execute(const NValueArray &params) {
    VOLT_DEBUG("executing NestLoop...");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(m_abstractNode);
    assert(node);
    assert(node->getInputTableCount() == 2);

    // output table must be a temp table
    assert(m_tmpOutputTable);

    Table* outer_table = node->getInputTable();
    assert(outer_table);

    Table* inner_table = node->getInputTable(1);
    assert(inner_table);

    VOLT_TRACE ("input table left:\n %s", outer_table->debug().c_str());
    VOLT_TRACE ("input table right:\n %s", inner_table->debug().c_str());

    //
    // Pre Join Expression
    //
    AbstractExpression *preJoinPredicate = node->getPreJoinPredicate();
    if (preJoinPredicate) {
        VOLT_TRACE ("Pre Join predicate: %s", preJoinPredicate == NULL ?
                    "NULL" : preJoinPredicate->debug(true).c_str());
    }
    //
    // Join Expression
    //
    AbstractExpression *joinPredicate = node->getJoinPredicate();
    if (joinPredicate) {
        VOLT_TRACE ("Join predicate: %s", joinPredicate == NULL ?
                    "NULL" : joinPredicate->debug(true).c_str());
    }
    //
    // Where Expression
    //
    AbstractExpression *wherePredicate = node->getWherePredicate();
    if (wherePredicate) {
        VOLT_TRACE ("Where predicate: %s", wherePredicate == NULL ?
                    "NULL" : wherePredicate->debug(true).c_str());
    }

    // Join type
    //JoinType join_type = node->getJoinType();
    //assert(join_type == JOIN_TYPE_INNER || join_type == JOIN_TYPE_LEFT);
    LimitPlanNode* limit_node = dynamic_cast<LimitPlanNode*>(node->getInlinePlanNode(PLAN_NODE_TYPE_LIMIT));
    int limit = -1;
    int offset = -1;
    if (limit_node) {
        limit_node->getLimitAndOffsetByReference(params, limit, offset);
    }

    int outer_cols = outer_table->columnCount();
    int inner_cols = inner_table->columnCount();
    TableTuple outer_tuple(node->getInputTable(0)->schema());
    TableTuple inner_tuple(node->getInputTable(1)->schema());
    //const TableTuple& null_tuple = m_null_tuple.tuple();

    TableIterator iterator0 = outer_table->iterator();
    TableIterator iterator1 = inner_table->iterator();
    //int tuple_ctr = 0;
    //int tuple_skipped = 0;
    ProgressMonitorProxy pmp(m_engine, this, inner_table);

    TableTuple join_tuple;
    if (m_aggExec != NULL) {
        VOLT_TRACE("Init inline aggregate...");
        const TupleSchema * aggInputSchema = node->getTupleSchemaPreAgg();
        join_tuple = m_aggExec->p_execute_init(params, &pmp, aggInputSchema, m_tmpOutputTable);
    } else {
        join_tuple = m_tmpOutputTable->tempTuple();
    }

    if(preJoinPredicate == NULL){
      printf("prejoin\n");
    }
    if(joinPredicate == NULL){
      printf("join\n");
    }
    if(wherePredicate == NULL){
      printf("where\n");
    }

    char *outer_GTT=NULL,*inner_GTT=NULL;
    GTupleSchema *GouterSchema = NULL,*GinnerSchema;

    TableTuple *tmpouter_tuple=NULL,*tmpinner_tuple=NULL;

    int outerSize = (int)outer_table->activeTupleCount();
    int innerSize = (int)inner_table->activeTupleCount();
    printf("leftsize:%d\trightsize:%d\n",outerSize,innerSize);


    makeExpressionTree etree;

    etree.maketree(joinPredicate,0,1);
    etree.setSize();
    printf("tree size : %d\n",etree.getSize());

    char *edata = (char *)calloc(1,etree.getSize());
    etree.allocate(joinPredicate,edata,0,1);

    EXPRESSIONNODE enode = etree.getENode(1);
    printf("enode data : %d %d %d %lu\n",static_cast<int>(enode.et),enode.startPos,enode.endPos,sizeof(GAbstractExpression));
    enode = etree.getENode(2);
    printf("enode data : %d %d %d %lu\n",static_cast<int>(enode.et),enode.startPos,enode.endPos,sizeof(GAbstractExpression));

    /*
    if(reinterpret_cast<GComparisonExpression*>(edata)->eval(NULL,NULL,NULL).isTrue()){
      printf("true\n");
    }else{
      printf("false\n");
    }
    */

    iterator0 = outer_table->iterator();
    iterator1 = inner_table->iterator();
    int j=0;

    /*
      all data must be transported to GPU.
     */

    /*
      TO DO : I must sratch a better implementation.
      TO DO : get all size of tabletuple included ColumnInfo etc.not outer_table->getTupleLength()
    */

    //move outer tuple
    int OschemaSize = static_cast<int>(sizeof(TupleSchema) + (outer_tuple.getSchema()->columnCount()+1)*sizeof(TupleSchema::ColumnInfo) + outer_tuple.getSchema()->getUninlinedObjectColumnCount()*sizeof(int16_t));
    GouterSchema = (GTupleSchema *)malloc(OschemaSize);
    memcpy(GouterSchema,outer_tuple.getSchema(),OschemaSize);
    printf("schema column count:%d %d %lu\n",outer_tuple.getSchema()->columnCount(),GouterSchema->columnCount(),sizeof(TupleSchema::ColumnInfo));

    tmpouter_tuple = (TableTuple *)malloc(outerSize*sizeof(TableTuple));
    int outerTupleSize = static_cast<int>(sizeof(GTableTuple)+outer_tuple.tupleLength()-1);
    outer_GTT = (char *)malloc(outerSize*outerTupleSize);
    while(iterator0.next(outer_tuple)){
      if(!outer_tuple.isActive()) continue;
      reinterpret_cast<GTableTuple*>(outer_GTT+j*outerTupleSize)->setRowNumber(j);
      //reinterpret_cast<GTableTuple*>(outer_GTT+j*outerTupleSize)->setSchemaSize(OschemaSize);
      //reinterpret_cast<GTableTuple*>(outer_GTT+j*outerTupleSize)->setSchema(Gouter_schema,OschemaSize);
      //copy except TUPLE HEADER
      reinterpret_cast<GTableTuple*>(outer_GTT+j*outerTupleSize)->setTuple(outer_tuple.address(),outer_tuple.tupleLength());
      tmpouter_tuple[j] = outer_tuple; 
      j++;
    }

    printf("outertable row num = %d %d\n",j,outerTupleSize);
    outerSize = j;

    j=0;
    int IschemaSize = static_cast<int>(sizeof(TupleSchema) + (inner_tuple.getSchema()->columnCount()+1)*sizeof(TupleSchema::ColumnInfo) + inner_tuple.getSchema()->getUninlinedObjectColumnCount()*sizeof(int16_t));
    GinnerSchema = (GTupleSchema *)malloc(IschemaSize);
    memcpy(GinnerSchema,inner_tuple.getSchema(),IschemaSize);
    
    tmpinner_tuple = (TableTuple *)malloc(innerSize*sizeof(TableTuple));
    int innerTupleSize = static_cast<int>(sizeof(GTableTuple)+inner_tuple.tupleLength()-1);
    inner_GTT = (char *)malloc(innerSize*innerTupleSize);
    while(iterator1.next(inner_tuple)){
      if(!inner_tuple.isActive()) continue;
      reinterpret_cast<GTableTuple*>(inner_GTT+j*innerTupleSize)->setRowNumber(j);
      //reinterpret_cast<GTableTuple*>(inner_GTT+j*innerTupleSize)->setSchemaSize(IschemaSize);
      //reinterpret_cast<GTableTuple*>(inner_GTT+j*innerTupleSize)->setSchema(Ginner_schema,IschemaSize);
      //copy except TUPLE HEADER
      reinterpret_cast<GTableTuple*>(inner_GTT+j*innerTupleSize)->setTuple(inner_tuple.address(),inner_tuple.tupleLength());
      tmpinner_tuple[j] = inner_tuple; 
      j++;
    }

    printf("innertable size = %d\n",j);        
    innerSize = j;



    bool gpuflag = true;

    //GPUNIJ *gn = new GPUNIJ();

    GPUNIJ *gn = new GPUNIJ();
    
    gpuflag = gn->initGPU();

    if(gpuflag){

      /*set table data: table size , GNValue ,expression*/
      gn->setTableData(outer_GTT,inner_GTT,outerSize,innerSize,outerTupleSize,innerTupleSize);
      gn->setSchema(GouterSchema,GinnerSchema,OschemaSize,IschemaSize);
      gn->setExpression(edata,etree.getSize());    

      RESULT *jt = NULL;
      int jt_size = 0;

      /*join
        return:
        matched left table and right table rows.             
        result table size 
      */

      if(gn->join()){
        jt = gn->getResult();
        jt_size = gn->getResultSize();
      }else{
        printf("join error.\n");
      }

      /* quick sort for debug */
      //qsort(jt,0,jt_size-1);
      /*insert result tuple*/
      for(int i=0; i < jt_size && (i<limit||limit==-1) ; i++){

        if (jt_size < offset) {
          continue;
        }


            
        join_tuple.setNValues(0, tmpouter_tuple[jt[i].lkey], 0, outer_cols);
        join_tuple.setNValues(outer_cols, tmpinner_tuple[jt[i].rkey], 0, inner_cols);

        if (m_aggExec != NULL) {
          if (m_aggExec->p_execute_tuple(join_tuple)) {
            break;
          }
        } else {
          m_tmpOutputTable->insertTempTuple(join_tuple);
        }


      }

    }else{
      printf("GPU init is false\n");          
    }
    gn->finish();
    delete gn;
    free(tmpouter_tuple);
    free(tmpinner_tuple);
    free(outer_GTT);
    free(inner_GTT);
    free(GinnerSchema);
    free(GouterSchema);
    free(edata);

    /*
    bool earlyReturned = false;
    while ((limit == -1 || tuple_ctr < limit) && iterator0.next(outer_tuple)){
        pmp.countdownProgress();

        // populate output table's temp tuple with outer table's values
        // probably have to do this at least once - avoid doing it many
        // times per outer tuple
        join_tuple.setNValues(0, outer_tuple, 0, outer_cols);

        // did this loop body find at least one match for this tuple?
        bool match = false;
        // For outer joins if outer tuple fails pre-join predicate
        // (join expression based on the outer table only)
        // it can't match any of inner tuples
        if (preJoinPredicate == NULL || preJoinPredicate->eval(&outer_tuple, NULL).isTrue()) {

            // By default, the delete as we go flag is false.
            TableIterator iterator1 = inner_table->iterator();
            while ((limit == -1 || tuple_ctr < limit) && iterator1.next(inner_tuple)) {
                pmp.countdownProgress();
                // Apply join filter to produce matches for each outer that has them,
                // then pad unmatched outers, then filter them all
                if (joinPredicate == NULL || joinPredicate->eval(&outer_tuple, &inner_tuple).isTrue()){
                    match = true;
                    // Filter the joined tuple
                    if (wherePredicate == NULL || wherePredicate->eval(&outer_tuple, &inner_tuple).isTrue()) {
                        // Check if we have to skip this tuple because of offset
                        if (tuple_skipped < offset) {
                            tuple_skipped++;
                            continue;
                        }
                        ++tuple_ctr;
                        // Matched! Complete the joined tuple with the inner column values.
                        join_tuple.setNValues(outer_cols, inner_tuple, 0, inner_cols);
                        if (m_aggExec != NULL) {
                            if (m_aggExec->p_execute_tuple(join_tuple)){
                                // Get enough rows for LIMIT
                                earlyReturned = true;
                                break;
                            }
                        } else {
                            m_tmpOutputTable->insertTempTuple(join_tuple);
                            pmp.countdownProgress();
                        }
                    }
                }
            } // END INNER WHILE LOOP
        } // END IF PRE JOIN CONDITION

        //
        // Left Outer Join
        //
        if (join_type == JOIN_TYPE_LEFT && !match && (limit == -1 || tuple_ctr < limit)) {
            // Still needs to pass the filter
            if (wherePredicate == NULL || wherePredicate->eval(&outer_tuple, &null_tuple).isTrue()) {
                // Check if we have to skip this tuple because of offset
                if (tuple_skipped < offset) {
                    tuple_skipped++;
                    continue;
                }
                ++tuple_ctr;
                join_tuple.setNValues(outer_cols, null_tuple, 0, inner_cols);
                if (m_aggExec != NULL) {
                    if (m_aggExec->p_execute_tuple(join_tuple)) {
                        earlyReturned = true;
                    }
                } else {
                    m_tmpOutputTable->insertTempTuple(join_tuple);
                    pmp.countdownProgress();
                }
            }
        } // END IF LEFT OUTER JOIN

        if (earlyReturned) {
            // Get enough rows for LIMIT inlined with aggregation
            break;
        }

    } // END OUTER WHILE LOOP
    */

    if (m_aggExec != NULL) {
        m_aggExec->p_execute_finish();
    }

    cleanupInputTempTable(inner_table);
    cleanupInputTempTable(outer_table);

    return (true);
}

