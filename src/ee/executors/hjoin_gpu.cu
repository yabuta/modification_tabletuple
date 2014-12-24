#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
//#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

extern "C" {

__global__
void count(
          COLUMNDATA *lt,
          COLUMNDATA *prt,
          uint *count,
          int *r_p,
          int p_num,
          int left
          ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;  
  if(x < left){

    GNValue tlgnv;

    if(x == left-1){
      tlgnv = lt[x].gn;
    }else{
      tlgnv = lt[x].gn;
    }

    uint temp = 0;
    int idx = tlgnv.getHashValue( 0 , p_num);
    int temp2 = r_p[idx+1];

    for(int k=r_p[idx] ; k<temp2 ; k++){
      /*
      if(ex.eval(tlgnv,prt[k].gn)){
        temp++;
      }
      */
    }

    count[x] = temp;

  }


  if(x == left-1){
    count[x+1] = 0;
  }

}


__global__ 
void join(
          COLUMNDATA *lt,
          COLUMNDATA *prt,
          RESULT *jt,
          int *r_p,
          uint *count,
          int p_num,
          int left
          ) 
{


  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < left){

    uint writeloc = count[x];

    GNValue tlgnv;
    if(x == left-1){
      tlgnv = lt[x].gn;
    }else{
      tlgnv = lt[x].gn;
    }

    int idx = tlgnv.getHashValue( 0 , p_num);
    int temp2 = r_p[idx+1];

    for(int k=r_p[idx] ; k<temp2 ; k ++){
      /*
      if(ex.eval(tlgnv,prt[k].gn)){
        jt[writeloc].lkey = lt[x].num;
        jt[writeloc].rkey = prt[k].num;
        writeloc++;
      }
      */
    }

  }

}    

}
