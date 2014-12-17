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


//#include <string>
//#include <iostream>
#include <stdio.h>

namespace voltdb {
    
    class GComparisonExpression : public GAbstractExpression{
        
    public:

        CUDAH GComparisonExpression():
        GAbstractExpression(),
            l_position(0),r_position(0),
            l_type(EXPRESSION_TYPE_INVALID),r_type(EXPRESSION_TYPE_INVALID)
        {};

        CUDAH GComparisonExpression(ExpressionType e,int l_pos, int r_pos,ExpressionType l_et,ExpressionType r_et) :
        GAbstractExpression(e),
            l_position(l_pos),r_position(r_pos),l_type(l_et),r_type(r_et)
        {
        };

        CUDAH GNValue eval(const GTableTuple *tuple1, const GTableTuple *tuple2, const char *data) const {

            assert(tuple1 != NULL && tuple2 != NULL);
            assert(data != NULL);

            GNValue NV1,NV2;

            switch(l_type){
            case EXPRESSION_TYPE_OPERATOR_PLUS:
            case EXPRESSION_TYPE_OPERATOR_MINUS:
            case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
            case EXPRESSION_TYPE_OPERATOR_DIVIDE:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_OPERATOR_NOT:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_OPERATOR_IS_NULL:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_COMPARE_EQUAL:
            case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
            case EXPRESSION_TYPE_COMPARE_LESSTHAN:
            case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
            case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
            case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
            {
                GComparisonExpression tmpGCE;
                memcpy(&tmpGCE,&data[l_position],sizeof(GComparisonExpression));
                NV1 = tmpGCE.eval(tuple1,tuple2,data);
                break;
            }
            case EXPRESSION_TYPE_CONJUNCTION_AND:
            case EXPRESSION_TYPE_CONJUNCTION_OR:
                return GNValue::getFalse();
                //return sizeof(ConjunctionExpression);
            case EXPRESSION_TYPE_VALUE_TUPLE:
            {
                GTupleValueExpression tmpGTVE;
                memcpy(&tmpGTVE,&data[l_position],sizeof(GTupleValueExpression));
                NV1 = tmpGTVE.eval(tuple1,tuple2,data);
                break;
            }
            default:
                return GNValue::getFalse();
            }


            switch(r_type){
            case EXPRESSION_TYPE_OPERATOR_PLUS:
            case EXPRESSION_TYPE_OPERATOR_MINUS:
            case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
            case EXPRESSION_TYPE_OPERATOR_DIVIDE:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_OPERATOR_NOT:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_OPERATOR_IS_NULL:
                return GNValue::getFalse();
            case EXPRESSION_TYPE_COMPARE_EQUAL:
            case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
            case EXPRESSION_TYPE_COMPARE_LESSTHAN:
            case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
            case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
            case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
            {
                GComparisonExpression tmpGCE;
                memcpy(&tmpGCE,&data[r_position],sizeof(GComparisonExpression));
                NV2 = tmpGCE.eval(tuple1,tuple2,data);
                break;
            }
            case EXPRESSION_TYPE_CONJUNCTION_AND:
            case EXPRESSION_TYPE_CONJUNCTION_OR:
                return GNValue::getFalse();
                //return sizeof(ConjunctionExpression);
            case EXPRESSION_TYPE_VALUE_TUPLE:
            {
                GTupleValueExpression tmpGTVE;
                memcpy(&tmpGTVE,&data[r_position],sizeof(GTupleValueExpression));
                NV2 = tmpGTVE.eval(tuple1,tuple2,data);
                break;
            }
            default:
                return GNValue::getFalse();
            }


            if(NV1.isNull()){
                return GNValue::getNullValue(VALUE_TYPE_BOOLEAN);
            }
            if(NV2.isNull()){
                return GNValue::getNullValue(VALUE_TYPE_BOOLEAN);
            }

            
            switch(m_type){
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
            return m_type;
        }


    private:

        int l_position;
        int r_position;
        ExpressionType l_type;
        ExpressionType r_type;


    };

}
#endif
