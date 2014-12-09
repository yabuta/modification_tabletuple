#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/common/GTupleSchema.h"
//#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

extern "C" {

  /**
     called function is changed by join condition.
     
     if T1.val = T2.val, iocount and iojoin is called.
     if T.val1 = T.val2 , iicount and iijoin is called.
   */


__global__
void count(
          GTableTuple *oGTT,
          GTableTuple *iGTT,
          char *od,
          char *id,
          GTupleSchema *os,
          GTupleSchema *is,
          int *count,
          int ltn,
          int rtn
          ) 

{


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){
    GTableTuple toGTT=oGTT[i];
    int rtn_g = rtn;
    int mcount = 0;

    printf("oGTT column count:%d\n",oGTT->getGNValue(0,os,od).getValueType());

    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){
      mcount++;
      /*
      if(ex.eval(toGTT,iGTT[BLOCK_SIZE_Y*blockIdx.y + j])) {
        mcount++;
      }     
      */
    }

    count[i+k] = mcount;
  }

  if(i+k == (blockDim.x*gridDim.x*gridDim.y-1)){
    count[i+k+1] = 0;
  }

}


__global__ void join(
          GTableTuple *oGTT,
          GTableTuple *iGTT,
          char *od,
          char *id,
          GTupleSchema *os,
          GTupleSchema *is,
          RESULT *p,
          int *count,
          int ltn,
          int rtn,
          int ll,
          int rr
          ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  if(i<ltn){

    GTableTuple toGTT = oGTT[i];
    int rtn_g = rtn;
    int writeloc = count[i+k];
    for(uint j = 0; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn_g;j++){
      p[writeloc].lkey = i;
      p[writeloc].rkey = BLOCK_SIZE_Y*blockIdx.y + j;
      writeloc++;
      /*
      if(ex.eval(toGTT,iGTT[BLOCK_SIZE_Y*blockIdx.y + j])){
        p[writeloc].lkey = i;
        p[writeloc].rkey = BLOCK_SIZE_Y*blockIdx.y + j;
        writeloc++;
      }
      */
    }
  } 

}

}
