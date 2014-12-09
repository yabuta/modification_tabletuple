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

using namespace voltdb;

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

#endif
