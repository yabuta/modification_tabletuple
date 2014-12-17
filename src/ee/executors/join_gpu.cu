#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/common/GTupleSchema.h"
#include "GPUetc/expressions/Gabstractexpression.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"
//#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

  /**
     called function is changed by join condition.
     
     if T1.val = T2.val, iocount and iojoin is called.
     if T.val1 = T.val2 , iicount and iijoin is called.
   */

extern "C"{

__global__
void count(
          char *oGTT,
          char *iGTT,
          int ots,
          int its,
          ulong *count,
          char *ex,
          GTupleSchema *os,
          GTupleSchema *is,
          int ossize,
          int issize,
          int ltn,
          int rtn,
          uint block_size_y,
          uint exSize
          ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  extern __shared__ char total[];
  char *tiGTT = total;
  GTupleSchema *tos = reinterpret_cast<GTupleSchema*>(&total[its*block_size_y]);
  GTupleSchema *tis = reinterpret_cast<GTupleSchema*>(&total[its*block_size_y+ossize]);
  for(int i = threadIdx.x; i<block_size_y && block_size_y*blockIdx.y+i<rtn ; i+=blockDim.x){
    memcpy(&tiGTT[i*its],iGTT + (block_size_y*blockIdx.y+i)*its,its);
  }

  if(threadIdx.x==0){
    memcpy(tos,os,ossize);
    memcpy(tis,is,issize);
  }

  if(x<ltn){
    //speedup step by storing to register
    char *toGTT = (char *)malloc(ots);
    memcpy(toGTT,oGTT+x*ots,ots);

    reinterpret_cast<GTableTuple*>(toGTT)->setSchema(tos);

    int rtn_g = rtn;
    int mcount = 0;

    for(uint y = 0; y<block_size_y && block_size_y*blockIdx.y+y<rtn_g;y++){
      reinterpret_cast<GTableTuple*>(tiGTT+y*its)->setSchema(tis);
      /*
      if(blockIdx.x==0&&threadIdx.x==0&&blockIdx.y==0){
        printf("tiGTT address:%d %d\n",reinterpret_cast<GTableTuple*>(tiGTT+y*its)->tupleLength(),its);
      }
      */
      if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(toGTT),reinterpret_cast<GTableTuple*>(tiGTT+y*its),ex).isTrue()) {
        //if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(toGTT),reinterpret_cast<GTableTuple*>(tiGTT+y*its),ex).isTrue()) {
        mcount++;
      }
    }

    count[x+k] = mcount;
    free(toGTT);

  }

  if(x+k == (blockDim.x*gridDim.x*gridDim.y-1)){
    count[x+k+1] = 0;
  }

}


__global__ void join(
          char *oGTT,
          char *iGTT,
          int ots,
          int its,
          RESULT *p,
          ulong *count,
          char *ex,
          GTupleSchema *os,
          GTupleSchema *is,
          int ossize,
          int issize,
          int ltn,
          int rtn,
          int ll,
          int rr,
          uint block_size_y,
          uint exSize
          ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  extern __shared__ char total[];
  char *tiGTT = total;
  GTupleSchema *tos = reinterpret_cast<GTupleSchema*>(&total[its*block_size_y]);
  GTupleSchema *tis = reinterpret_cast<GTupleSchema*>(&total[its*block_size_y+ossize]);

  for(int i = threadIdx.x; i<block_size_y && block_size_y*blockIdx.y+i<rtn ; i+=blockDim.x){
    memcpy(&tiGTT[i*its],iGTT + (block_size_y*blockIdx.y+i)*its,its);
  }

  if(threadIdx.x==0){
    memcpy(tos,os,ossize);
    memcpy(tis,is,issize);
  }

  if(x<ltn){

    //speedup step by storing to register
    char *toGTT = (char *)malloc(ots);
    memcpy(toGTT,oGTT+x*ots,ots);

    reinterpret_cast<GTableTuple*>(toGTT)->setSchema(tos);

    int rtn_g = rtn;
    int writeloc = count[x+k];

    for(uint y = 0; y<block_size_y && block_size_y*blockIdx.y+y<rtn_g;y++){
      reinterpret_cast<GTableTuple*>(tiGTT+y*its)->setSchema(tis);
      if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(toGTT),reinterpret_cast<GTableTuple*>(tiGTT+y*its),ex).isTrue()) {
        //if(tex->eval(reinterpret_cast<GTableTuple*>(toGTT),reinterpret_cast<GTableTuple*>(tiGTT+y*its),ex).isTrue()) {
        p[writeloc].lkey = x;
        p[writeloc].rkey = block_size_y*blockIdx.y + y;
        writeloc++;
      } 
    }

    free(toGTT);
  }

}

}
