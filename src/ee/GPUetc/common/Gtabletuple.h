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

#ifndef GTABLETUPLE_H
#define GTABLETUPLE_H


//#include "common/common.h"
//#include "common/TupleSchema.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/GTupleSchema.h"
//#include "common/Pool.hpp"
//#include "common/ValuePeeker.hpp"
//#include "common/FatalException.hpp"
//#include "common/ExportSerializeIo.h"

#include <cassert>

namespace voltdb {

#define TUPLE_HEADER_SIZE 1

class GTableTuple {

public:
    /** Initialize a tuple unassociated with a table (bad idea... dangerous) */
    CUDAH GTableTuple(){};
    CUDAH ~GTableTuple(){};

    /** Get the value of a specified column (const) */
    //not performant because it has to check the schema to see how to
    //return the SlimValue.
    inline CUDAH const GNValue getGNValue(const int idx,const GTupleSchema *schema,const char *start) const {
        assert(schema);
        assert(start);
        assert(idx < schema->columnCount());

        const GTupleSchema::ColumnInfo *columnInfo = schema->getColumnInfo(idx);
        const voltdb::ValueType columnType = columnInfo->getVoltType();
        const char* dataPtr = getDataPtr(columnInfo,schema,start);
        const bool isInlined = columnInfo->inlined;

        return GNValue::initFromTupleStorage(dataPtr, columnType, isInlined);
    }

    /** How long is a tuple? */
    inline CUDAH int tupleLength(const GTupleSchema *schema) const {
        return schema->tupleLength() + TUPLE_HEADER_SIZE;
    }

    CUDAH void setRowNumber(int rn){
        rownumber = rn;
    }

private:

    inline CUDAH const char* getDataPtr(const GTupleSchema::ColumnInfo * colInfo, const GTupleSchema *schema,const char *start) const {
        assert(schema);
        assert(start);
        return &start[TUPLE_HEADER_SIZE + colInfo->offset + (rownumber-1)*tupleLength(schema)];
    }


    /**
     *the row number of this tuple.
     */
    int rownumber;

};

} // namespace voltdb

#endif