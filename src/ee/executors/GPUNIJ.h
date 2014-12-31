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

    void setTableData(char *ogtt,char *igtt,int outerSize,int innerSize,int ots,int its) {
        
        assert(outerSize >= 0 && innerSize >= 0);
        assert(ogtt != NULL && igtt != NULL);

        outer_GTT = ogtt;
        inner_GTT = igtt;
        left = outerSize;
        right = innerSize;
        outerTupleSize = ots;
        innerTupleSize = its;

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

    inline void setSchema(GTupleSchema *os,GTupleSchema *is,int ossize,int issize){
        assert(ossize >= 0 && issize >= 0);
        assert(os != NULL && is != NULL);
        outerSchema = os;
        innerSchema = is;
        outerSchemaSize = ossize;
        innerSchemaSize = issize;
    }

    inline void setExpression(char *edata,int size){
        assert(size >= 0);
        expression = edata;
        exSize = size;
    }

    RESULT *getResult() const {
        return jt;
    }

    int getResultSize() const {
        return total;
    }


private:

//for partition execution
   
    RESULT *jt;
    int total;

    uint left,right;
    char *outer_GTT;
    char *inner_GTT;
    int outerTupleSize;
    int innerTupleSize;

    GTupleSchema *outerSchema;
    GTupleSchema *innerSchema;
    int outerSchemaSize;
    int innerSchemaSize;

    char *expression;
    int exSize;

    int PART;

    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function;
    CUmodule module,c_module;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};



}

#endif
