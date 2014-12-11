/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H
#define GPUNIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/common/GTupleSchema.h"

#include "expressions/abstractexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "expressions/comparisonexpression.h"
#include "GPUetc/expressions/Gabstractexpression.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"
#include "GPUetc/expressions/Gtuplevalueexpression.h"
#include "GPUetc/expressions/nodedata.h"


namespace voltdb{

class GPUNIJ{

public:

    GPUNIJ();

    bool initGPU();
    void finish();
    bool join();


/**
   outer tuple = left
   inner tuple = right
 */

    void setTableData(GTableTuple *ogtt,GTableTuple *igtt,char *od,char *id,GTupleSchema *os,GTupleSchema *is,int outerSize,int innerSize){
        
        assert(outerSize >= 0 && innerSize >= 0);
        assert(ogtt != NULL && igtt != NULL);
        assert(od != NULL && id != NULL);
        assert(os != NULL && is != NULL);

        outer_GTT = ogtt;
        inner_GTT = igtt;
        outer_data = od;
        inner_data = id;
        outer_schema = os;
        inner_schema = is;

        left = outerSize;
        right = innerSize;

        PART = 262144;
        
        uint biggerTupleSize = left;
        if(left < right) biggerTupleSize = right;

        for(int i=32768 ; i<=262144 ; i = i*2){
            if(biggerTupleSize<=i){
                PART = i;
                break;
            }
        }
        printf("PART : %d\n",PART);

    }

    RESULT *getResult(){
        return jt;
    }

    int getResultSize(){
        return total;
    }


private:

//for partition execution
   
    RESULT *jt;
    int total;

    uint left,right;
    GTableTuple *outer_GTT;
    GTableTuple *inner_GTT;
    char *outer_data;
    char *inner_data;
    GTupleSchema *outer_schema;
    GTupleSchema *inner_schema;

    int PART;

    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function;
    CUmodule module,c_module;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};




/**
This class make a tree including expression information for GPGPU.
Each node has ExpressionType, startPos and etc.
The reason is that expression data is integrated in one array because we must send necessary data to GPU.
So I make one array including Expressions. 
The startPos of the structure of this class is the start position of an Expression in the array of GPU.

I assume this class used to seach an Expression in the GPU array.

*/

class makeExpressionTree{

public:

  makeExpressionTree();

  /*
    reclusive function. first is maketree(ab,0,1)
   */

  /*
  void maketree(const AbstractExpression *ab,int pos,int dep){

    if(ab == NULL){
        if(pos == 0){
            enode[pos].et = EXPRESSION_TYPE_INVALID;
            enode[pos].startPos = 0;
            enode[pos].endPos = 0;
            
        }else{
            enode[pos].et = EXPRESSION_TYPE_INVALID;
            enode[pos].startPos = enode[pos-1].startPos;
            enode[pos].endPos = enode[pos-1].endPos;
        }
    }
    
    enode[pos].et = ab->getExpressionType();
    if(enode[pos].et != EXPRESSION_TYPE_VALUE_TUPLE){

      if(pos == 0){
        enode[pos].startPos = 0;
        enode[pos].endPos = expressionSize(enode[pos].et);
      }else{
        enode[pos].startPos = enode[pos-1].startPos + expressionSize(enode[pos-1].et); 
        enode[pos].endPos = enode[pos].startPos + expressionSize(enode[pos].et);
      }

    }

    //bother to include math.h
    int nextdep = 1;
    for(int i=0 ; i<dep; i++){
      nextdep *= 2;
    }

    int nextpos = (nextdep-1) + (pos+1-nextdep/2)*2 + 1;

    if(enode[pos].et == EXPRESSION_TYPE_VALUE_TUPLE || ab == NULL){
      maketree(NULL,nextpos,dep+1);
      maketree(NULL,nextpos+1,dep+1);
    }else{
      maketree(ab->getLeft(),nextpos,dep+1);
      maketree(ab->getRight(),nextpos+1,dep+1);

    }
      
  }

  void allocate(const AbstractExpression *ab, char *data,int pos, int dep) {


      switch(enode[pos].et){
      case EXPRESSION_TYPE_OPERATOR_PLUS:
      case EXPRESSION_TYPE_OPERATOR_MINUS:
      case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
      case EXPRESSION_TYPE_OPERATOR_DIVIDE:
      {
          GComparisonExpression *tmpGCE = new GComparisonExpression(enode[pos].et,pos,dep);
          tmpGCE->setInBytes(ab->getInBytes());
          tmpGCE->setValueSize(ab->getValueSize());
          tmpGCE->setValueType(ab->getValueType());
          memcpy(&data[enode[pos].startPos],tmpGCE,sizeof(GComparisonExpression));
          break;
      }
      case EXPRESSION_TYPE_OPERATOR_NOT:
          break;
      case EXPRESSION_TYPE_OPERATOR_IS_NULL:
          break;
      case EXPRESSION_TYPE_COMPARE_EQUAL:
      case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
      case EXPRESSION_TYPE_COMPARE_LESSTHAN:
      case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
      case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
      case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
          break;
      case EXPRESSION_TYPE_CONJUNCTION_AND:
      case EXPRESSION_TYPE_CONJUNCTION_OR:
          break;
      case EXPRESSION_TYPE_VALUE_TUPLE:
      {
          TupleValueExpression *tmpTV = reinterpret_cast<TupleValueExpression*>(const_cast<AbstractExpression*>(ab));
          GTupleValueExpression *tmpGTVE = new GTupleValueExpression(tmpTV->getTupleId(),tmpTV->getColumnId(),pos,dep);
          tmpGTVE->setInBytes(ab->getInBytes());
          tmpGTVE->setValueSize(ab->getValueSize());
          tmpGTVE->setValueType(ab->getValueType());
          memcpy(&data[enode[pos].startPos],tmpGTVE,sizeof(GTupleValueExpression));
          break;
      }
      default:
          break;
      }

      //bother to include math.h
      int nextdep = 1;
      for(int i=0 ; i<dep; i++){
          nextdep *= 2;
      }
      
      int nextpos = (nextdep-1) + (pos+1-nextdep/2)*2 + 1;
      
      if(enode[pos].et == EXPRESSION_TYPE_VALUE_TUPLE || ab == NULL){
          allocate(NULL ,data ,nextpos ,dep+1);
          allocate(NULL ,data ,nextpos+1 ,dep+1);
      }else{
          allocate(ab->getLeft() ,data ,nextpos ,dep+1);
          allocate(ab->getRight() ,data ,nextpos+1 ,dep+1);
      }
      
  }
  

  int getSize(){
      return enode[14].endPos;
  }

  EXPRESSIONNODE getENode(int idx){
    assert(idx < 16);
    return enode[idx];
  }


private:


  int expressionSize(ExpressionType et){

    switch(et){
    case EXPRESSION_TYPE_OPERATOR_PLUS:
    case EXPRESSION_TYPE_OPERATOR_MINUS:
    case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
    case EXPRESSION_TYPE_OPERATOR_DIVIDE:
        return 0;
        //return sizeof(OperatorExpression);
    case EXPRESSION_TYPE_OPERATOR_NOT:
        return 0;
        //return sizeof(OperatorNOTExpression);
    case EXPRESSION_TYPE_OPERATOR_IS_NULL:
        return 0;
        //return sizeof(OperatorIsNullExpression);
    case EXPRESSION_TYPE_COMPARE_EQUAL:
    case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
    case EXPRESSION_TYPE_COMPARE_LESSTHAN:
    case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
    case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
    case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
        return sizeof(GComparisonExpression);
    case EXPRESSION_TYPE_CONJUNCTION_AND:
    case EXPRESSION_TYPE_CONJUNCTION_OR:
        return 0;
        //return sizeof(ConjunctionExpression);
    case EXPRESSION_TYPE_VALUE_TUPLE:
      return sizeof(GTupleValueExpression);
    default:
      return 0;     
    }
  }

  
  EXPRESSIONNODE enode[15];
  */  

};

}

#endif
