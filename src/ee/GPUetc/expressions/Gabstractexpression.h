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

#ifndef GABSTRACTEXPRESSION_H
#define GABSTRACTEXPRESSION_H

#include "common/types.h"
#include "common/PlannerDomValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"

#include <string>
#include <vector>

namespace voltdb {

/**
 * Predicate objects for filtering tuples during query execution.
 */

// ------------------------------------------------------------------
// AbstractExpression
// Base class for all expression nodes
// ------------------------------------------------------------------
class GAbstractExpression {
  public:
    /** destroy this node and all children */
    CUDAH virtual ~GAbstractExpression();

    CUDAH virtual GNValue eval(const GTableTuple *tuple1 = NULL, const GTableTuple *tuple2 = NULL,const EXPRESSIONNODE *enode = NULL,const char *data = NULL) const = 0;
/*
    virtual NValue getLeftNV(const TableTuple *tuple1 = NULL, const TableTuple *tuple2 = NULL) const=0;
    virtual NValue getRightNV(const TableTuple *tuple1 = NULL, const TableTuple *tuple2 = NULL) const=0;
    virtual int getLeftTupleId();
    virtual int getRightTupleId();
    virtual int getTupleId();
*/

    /** accessors */
    CUDAH ExpressionType getExpressionType() const {
        return m_type;
    }

    CUDAH ValueType getValueType() const
    {
        return m_valueType;
    }

    CUDAH int getValueSize() const
    {
        return m_valueSize;
    }

    CUDAH bool getInBytes() const
    {
        return m_inBytes;
    }

    // These should really be part of the constructor, but plumbing
    // the type and size args through the whole of the expression world is
    // not something I'm doing right now.
    CUDAH void setValueType(ValueType type)
    {
        m_valueType = type;
    }

    CUDAH void setInBytes(bool bytes)
    {
        m_inBytes = bytes;
    }

    CUDAH void setValueSize(int size)
    {
        m_valueSize = size;
    }

  protected:
    CUDAH GAbstractExpression();
    CUDAH GAbstractExpression(ExpressionType type);
    CUDAH GAbstractExpression(ExpressionType type,
                              int pos,int dep);

  protected:
    ExpressionType m_type;
    bool m_hasParameter;
    ValueType m_valueType;
    int m_valueSize;
    bool m_inBytes;
    int m_pos;
    int m_dep;

};


// ------------------------------------------------------------------
// AbstractExpression
// ------------------------------------------------------------------
CUDAH GAbstractExpression::GAbstractExpression()
    : m_type(EXPRESSION_TYPE_INVALID),
    m_hasParameter(true),m_pos(0),m_dep(0)
    {
    }

CUDAH GAbstractExpression::GAbstractExpression(ExpressionType type)
    : m_type(type),
    m_hasParameter(true),m_pos(0),m_dep(0)
    {
    }

CUDAH GAbstractExpression::GAbstractExpression(ExpressionType type,
                                               int pos,
                                               int dep
    )
    : m_type(type),
    m_hasParameter(true),m_pos(pos),m_dep(dep)
    {
    }


}
#endif
