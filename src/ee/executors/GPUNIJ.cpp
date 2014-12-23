#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "GPUNIJ.h"
#include "scan_common.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/common/GTupleSchema.h"

#define SHARED_SIZE 48 * 1024
#define GRID_Y_MAX 65536 //the max number of block including in grid.

using namespace voltdb;

GPUNIJ::GPUNIJ(){

  jt = NULL;
  outer_GTT = NULL;
  inner_GTT = NULL;
  total = 0;

}


bool GPUNIJ::initGPU(){ 

  char fname[256];
  char *vd;
  char path[256];
  //char *path="/home/yabuta/voltdb/voltdb";//TODO : get voltdb/voltdb path
    
  if((vd = getenv("VOLT_HOME")) != NULL){
    snprintf(path,256,"%s/voltdb/voltdb",vd);
  }else if((vd = getenv("HOME")) != NULL){
    snprintf(path,256,"%s/voltdb/voltdb",vd);
  }else{
    return false;
  }

  /******************** GPU init here ************************************************/
  //GPU仕様のために

  /*
  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  */

  /*********************************************************************************/


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  
  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(join) failed res=%lu\n",(unsigned long)res);
    return false;
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(join) failed\n");
    return false;
  }
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count) failed\n");
    return false;
  }

  return true;
}


void GPUNIJ::finish(){

  if(jt!=NULL){
    free(jt);
  }
  //finish GPU   ****************************************************

  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
  }  

  /*
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
  }
  */
}



void
GPUNIJ::printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

uint GPUNIJ::iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
bool GPUNIJ::join()
{

  //int i, j;
  uint gpu_size;
  ulong jt_size;
  CUdeviceptr lt_dev, rt_dev; 
  CUdeviceptr os_dev,is_dev,expression_dev;
  CUdeviceptr jt_dev,count_dev;
  unsigned int block_x, block_y, grid_x, grid_y,block_size_y;
  struct timeval count_s,count_f,join_s,join_f;


  /************** block_x * block_y is decided by BLOCK_SIZE. **************/


  //The number of tuple which is executed in 1 thread. 
  block_size_y = ((SHARED_SIZE/4)-outerSchemaSize-innerSchemaSize)/innerTupleSize;
  double temp = block_size_y;
  if(temp < 2){
    block_size_y = 1;
  }else if(floor(log2(temp))==ceil(log2(temp))){
    block_size_y = static_cast<int>(temp);
  }else{
    block_size_y = pow(2,static_cast<int>(log2(temp)) + 1);
  }

  if(block_y*GRID_Y_MAX < right){
    printf("A block_y is too small.\nMaybe right table tuple length is too big.\n");
    return false;
  }


  block_x = BLOCK_SIZE_X;
  block_y = block_size_y;

    grid_x = PART / block_x;
  if (PART % block_x != 0)
    grid_x++;
  grid_y = PART / block_y;
  if (PART % block_y != 0)
    grid_y++;
  block_y = 1;

  gpu_size = grid_x * grid_y * block_x * block_y+1;
  if(gpu_size>MAX_LARGE_ARRAY_SIZE){
    gpu_size = MAX_LARGE_ARRAY_SIZE * iDivUp(gpu_size,MAX_LARGE_ARRAY_SIZE);
  }else if(gpu_size > MAX_SHORT_ARRAY_SIZE){
    gpu_size = MAX_SHORT_ARRAY_SIZE * iDivUp(gpu_size,MAX_SHORT_ARRAY_SIZE);
  }else{
    gpu_size = MAX_SHORT_ARRAY_SIZE;
  }


  /********************************************************************************/

  //tuple class data
  res = cuMemAlloc(&lt_dev, PART * outerTupleSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    return false;
  }
  res = cuMemAlloc(&rt_dev, PART * innerTupleSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    return false;
  }

  //outer schema data
  res = cuMemAlloc(&os_dev, innerSchemaSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (os) failed\n");
    return false;
  }
  res = cuMemcpyHtoD(os_dev, outerSchema, outerSchemaSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (os) failed: res = %lu\n", res);//conv(res));
    return false;
  }
  //inner schema data
  res = cuMemAlloc(&is_dev, innerSchemaSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (is) failed\n");
    return false;
  }
  res = cuMemcpyHtoD(is_dev, innerSchema, innerSchemaSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (is) failed: res = %lu\n", res);//conv(res));
    return false;
  }


  //expression data
  res = cuMemAlloc(&expression_dev, exSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (expression) failed\n");
    return false;
  }
  res = cuMemcpyHtoD(expression_dev, expression, exSize);
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (expression) failed: res = %lu\n", res);//conv(res));
    return false;
  }


  res = cuMemAlloc(&count_dev, gpu_size * sizeof(ulong));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    return false;
  }

  /********************** upload lt , rt and count***********************/

  for(uint ll = 0; ll < left ; ll += PART){
    for(uint rr = 0; rr < right ; rr += PART){


      gettimeofday(&count_s,NULL);

      uint lls=PART,rrs=PART;
      if((ll+PART) >= left){
        lls = left - ll;
      }
      if((rr+PART) >= right){
        rrs = right - rr;
      }

      block_x = lls < BLOCK_SIZE_X ? lls : BLOCK_SIZE_X;
      block_y = rrs < block_size_y ? rrs : block_size_y;      
      grid_x = lls / block_x;
      if (lls % block_x != 0)
        grid_x++;
      grid_y = rrs / block_y;
      if (rrs % block_y != 0)
        grid_y++;
      printf("grid_x = %d\tgrid_y = %d\tblock_x = %d\tblock_y = %d\n",grid_x,grid_y,block_x,block_y);

      block_y = 1;

      printf("\nStarting...\nll = %d\trr = %d\tlls = %d\trrs = %d\n",ll,rr,lls,rrs);
      gpu_size = grid_x * grid_y * block_x * block_y + 1;
      printf("gpu_size = %d\n",gpu_size);


      res = cuMemcpyHtoD(lt_dev, &(outer_GTT[ll]), lls * outerTupleSize);
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
        return false;
      }
      res = cuMemcpyHtoD(rt_dev, &(inner_GTT[rr]), rrs * innerTupleSize);
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
        return false;
      }

      void *count_args[]={
        (void *)&lt_dev,
        (void *)&rt_dev,
        (void *)&outerTupleSize,
        (void *)&innerTupleSize,
        (void *)&count_dev,
        (void *)&expression_dev,
        (void *)&os_dev,
        (void *)&is_dev,
        (void *)&outerSchemaSize,
        (void *)&innerSchemaSize,
        (void *)&lls,
        (void *)&rrs,
        (void *)&block_size_y
      };
      
      res = cuLaunchKernel(
                           c_function,    // CUfunction f
                           grid_x,        // gridDimX
                           grid_y,        // gridDimY
                           1,             // gridDimZ
                           block_x,       // blockDimX
                           block_y,       // blockDimY
                           1,             // blockDimZ
                           innerTupleSize * block_size_y 
                           + outerSchemaSize 
                           + innerSchemaSize,
                           // sharedMemBytes
                           NULL,          // hStream
                           count_args,   // kernelParams
                           NULL           // extra
                           );
      if(res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
        return false;
      }      
      
      res = cuCtxSynchronize();
      if(res != CUDA_SUCCESS) {
        printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
        return false;
      }  

      /**************************** prefix sum *************************************/
      if(!((new GPUSCAN<ulong,ulong4>)->presum(&count_dev,gpu_size))){
        printf("count scan error.\n");
        return false;
      }
      /********************************************************************/      

      if(!(new GPUSCAN<ulong,ulong4>)->getValue(count_dev,gpu_size,&jt_size)){
        printf("transport error.\n");
        return false;
      }

      gettimeofday(&count_f,NULL);

      /************************************************************************
      jt memory alloc and jt upload

      ************************************************************************/

      printf("jt_size %d\n",jt_size);

      gettimeofday(&join_s,NULL);

      if(jt_size <0){
        return false;
      }else if(jt_size > 64*1024*1024){
        printf("one time result size is over.\n");
        return true;

      }else if(total > 1024*1024*1024){
        printf("result size is over.\n");
        return true;
      }else if(jt_size==0){
        total += jt_size;
        jt_size = 0;
      }else{
        jt = (RESULT *)realloc(jt,(total+jt_size)*sizeof(RESULT));
        res = cuMemAlloc(&jt_dev, jt_size*sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemAlloc (join) failed\n");
          return false;
        }      

        void *kernel_args[]={
          (void *)&lt_dev,
          (void *)&rt_dev,
          (void *)&outerTupleSize,
          (void *)&innerTupleSize,
          (void *)&jt_dev,
          (void *)&count_dev,
          (void *)&expression_dev,
          (void *)&os_dev,
          (void *)&is_dev,
          (void *)&outerSchemaSize,
          (void *)&innerSchemaSize,
          (void *)&lls,
          (void *)&rrs,    
          (void *)&ll,
          (void *)&rr,
          (void *)&block_size_y
        };

        res = cuLaunchKernel(
                             function,      // CUfunction f
                             grid_x,        // gridDimX
                             grid_y,        // gridDimY
                             1,             // gridDimZ
                             block_x,       // blockDimX
                             block_y,       // blockDimY
                             1,             // blockDimZ
                             innerTupleSize * block_size_y 
                             + outerSchemaSize
                             + innerSchemaSize,
                             // sharedMemBytes
                             NULL,          // hStream
                             kernel_args,   // keunelParams
                             NULL           // extra
                             );
        if(res != CUDA_SUCCESS) {
          printf("cuLaunchKernel(join) failed: res = %lu\n", (unsigned long int)res);
          return false;
        }  
        
        res = cuCtxSynchronize();
        if(res != CUDA_SUCCESS) {
          printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
          return false;
        }  

          
        res = cuMemcpyDtoH(&(jt[total]), jt_dev, jt_size * sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
          return false;
        }
        cuMemFree(jt_dev);
        total += jt_size;
        printf("End...\n jt_size = %d\ttotal = %d\n",jt_size,total);
        jt_size = 0;

        gettimeofday(&join_f,NULL);
      
        
      }
    }
    

  }


  printDiff(count_s,count_f);
  printDiff(join_s,join_f);

  /***************************************************************/

  //free GPU memory***********************************************


  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(os_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (os) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(is_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (is) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(expression_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (expression) failed: res = %lu\n", (unsigned long)res);
    return false;
  }

  return true;

}


