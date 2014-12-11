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

#ifndef GCOMPARISONEXPRESSION_H
#define GCOMPARISONEXPRESSION_H

/*
#include "common/common.h"
#include "common/serializeio.h"
#include "common/valuevector.h"

#include "expressions/abstractexpression.h"
#include "expressions/parametervalueexpression.h"
#include "expressions/constantvalueexpression.h"
#include "expressions/tuplevalueexpression.h"
*/

#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"
#include "GPUetc/expressions/Gabstractexpression.h"
#include "GPUetc/expressions/Gtuplevalueexpression.h"


#include <string>
#include <iostream>
#include <stdio.h>

namespace voltdb {

class GComparisonExpression : public GAbstractExpression{

public:

    GComparisonExpression();

    CUDAH GComparisonExpression(ExpressionType e, int pos, int dep)  : GAbstractExpression(e,pos,dep), et(e)
    {
    };

    CUDAH virtual GNValue eval(const GTableTuple *tuple1, const GTableTuple *tuple2, const EXPRESSIONNODE *enode,const char *data) const {

        //bother to include math.h
        int nextdep = 1;
        for(int i=0 ; i<m_dep; i++){
            nextdep *= 2;
        }

        int nextpos = (nextdep-1) + (m_pos+1-nextdep/2)*2 + 1;

        GAbstractExpression *tmp = expressionGetter(&data[enode[nextpos].startPos]);
        GNValue NV1 = tmp->eval(tuple1,tuple2,enode,data);
        if(NV1.isNull()){
            return GNValue::getNullValue(VALUE_TYPE_BOOLEAN);
        }
        tmp = expressionGetter(&data[enode[nextpos+1].startPos]);
        GNValue NV2 = tmp->eval(tuple1,tuple2,enode,data);
        if(NV2.isNull()){
            return GNValue::getNullValue(VALUE_TYPE_BOOLEAN);
        }

        switch(et){
        case (EXPRESSION_TYPE_COMPARE_EQUAL):
            return NV1.op_equals_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_NOTEQUAL):
            return NV1.op_notEquals_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_LESSTHAN):
            return NV1.op_lessThan_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_GREATERTHAN):
            return NV1.op_greaterThan_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO):
            return NV1.op_lessThanOrEqual_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO):
            return NV1.op_greaterThanOrEqual_withoutNull(NV2);
        case (EXPRESSION_TYPE_INVALID):
            return GNValue::getTrue();
        default:
            return GNValue::getFalse();
        }
    }
    
    CUDAH int getET(){
        return et;
    }


private:

    CUDAH GAbstractExpression *expressionGetter(const char *data) const {

        switch(et){
        case EXPRESSION_TYPE_OPERATOR_PLUS:
        case EXPRESSION_TYPE_OPERATOR_MINUS:
        case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
        case EXPRESSION_TYPE_OPERATOR_DIVIDE:
            return NULL;
            //return sizeof(OperatorExpression);
        case EXPRESSION_TYPE_OPERATOR_NOT:
            return NULL;
            //return sizeof(OperatorNOTExpression);
        case EXPRESSION_TYPE_OPERATOR_IS_NULL:
            return NULL;
            //return sizeof(OperatorIsNullExpression);
        case EXPRESSION_TYPE_COMPARE_EQUAL:
        case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
        case EXPRESSION_TYPE_COMPARE_LESSTHAN:
        case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
        case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
        case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
        {
            GComparisonExpression *tmpGCE = NULL;
            memcpy(tmpGCE,data,sizeof(GComparisonExpression));
            return static_cast<GAbstractExpression*>(tmpGCE);
        }
        case EXPRESSION_TYPE_CONJUNCTION_AND:
        case EXPRESSION_TYPE_CONJUNCTION_OR:
            return NULL;
            //return sizeof(ConjunctionExpression);
        case EXPRESSION_TYPE_VALUE_TUPLE:
        {
            GTupleValueExpression *tmpGTVE = NULL;
            memcpy(tmpGTVE,data,sizeof(GTupleValueExpression));
            return static_cast<GAbstractExpression*>(tmpGTVE);
        }
        default:
            return NULL;
        }
    }


    ExpressionType et;


};

}
#endif
