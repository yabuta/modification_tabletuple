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
           char *oGTT, //outer GTableTuple
           char *iGTT, //inner GTableTuple
           int ots, //outer tuple size
           int its, //inner tuple size
           ulong *count,
           char *ex, // join predicate
           GTupleSchema *os,
           GTupleSchema *is,
           int ossize,
           int issize,
           int ltn,
           int rtn,
           uint block_size_y
           ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  //speedup by storing to shared memory
  extern __shared__ char total[];
  char *tiGTT = total;
  //char *toGTT = &total[its*block_size_y];
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
    /*
    char *toGTT = (char *)malloc(ots);
    memcpy(toGTT,oGTT+x*ots,ots);
    */    
    //reinterpret_cast<GTableTuple*>(toGTT)->setSchema(tos);

    int rtn_g = rtn;
    int mcount = 0;
    for(uint y = 0; y<block_size_y && block_size_y*blockIdx.y+y<rtn_g;y++){
      /*
      if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(toGTT),
                                                            reinterpret_cast<GTableTuple*>(tiGTT+y*its),
                                                            ex,
                                                            tos,
                                                            tis)) {
      */
        //if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(toGTT),reinterpret_cast<GTableTuple*>(tiGTT+y*its),ex).isTrue()) {
      if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(oGTT+x*ots),
                                                            reinterpret_cast<GTableTuple*>(tiGTT+y*its),
                                                            ex,
                                                            tos,
                                                            tis)) {

        mcount++;
      }

    }

    count[x+k] = mcount;
    //free(toGTT);
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
          uint block_size_y
          ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;


  //speedup by storing to shared memory
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
    /*
    char *toGTT = (char *)malloc(ots);
    memcpy(toGTT,oGTT+x*ots,ots);
    */
    int rtn_g = rtn;
    uint writeloc = count[x+k];
    for(uint y = 0; y<block_size_y && block_size_y*blockIdx.y+y<rtn_g;y++){
      if(reinterpret_cast<GComparisonExpression*>(ex)->eval(reinterpret_cast<GTableTuple*>(oGTT+x*ots),
                                                            reinterpret_cast<GTableTuple*>(tiGTT+y*its),
                                                            ex,
                                                            tos,
                                                            tis)) {
        p[writeloc].lkey = reinterpret_cast<GTableTuple*>(oGTT+x*ots)->getRowNumber();
        p[writeloc].rkey = reinterpret_cast<GTableTuple*>(tiGTT+y*its)->getRowNumber();
        writeloc++;
      }

    }

    //free(toGTT);
  }

}

}
