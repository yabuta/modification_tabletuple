#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
//#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "debug.h"
#include "scan_common.h"
#include "tuple.h"

TUPLE *rt;
TUPLE *lt;
RESULT *jt;


void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

static uint iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


static int
getTupleId(void)
{
  static int id;
  return ++id;
}

void shuffle(TUPLE ary[],int size) {    
  for(int i=0;i<size;i++){
    int j = rand()%size;
    TUPLE t = ary[i];
    ary[i] = ary[j];
    ary[j] = t;
  }
}


void createTuple()
{

  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て
  CUresult res;

  //メモリ割り当てを行う
  //タプルに初期値を代入

  //RIGHT_TUPLEへのGPUでも参照できるメモリの割り当て****************************
  res = cuMemHostAlloc((void**)&rt,right * sizeof(TUPLE),CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to RIGHT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }


  srand((unsigned)time(NULL));
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < right; i++) {
    if(&(rt[i])==NULL){
      printf("right TUPLE allocate error.\n");
      exit(1);
    }
    rt[i].key = getTupleId();
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    rt[i].val = temp2; 
  }


  //LEFT_TUPLEへのGPUでも参照できるメモリの割り当て*******************************
  res = cuMemHostAlloc((void**)&lt,left * sizeof(TUPLE),CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  uint counter = 0;//matchするtupleをcountする。
  uint *used_r;
  used_r = (uint *)calloc(right,sizeof(uint));
  for(uint i=0; i<right ; i++){
    used_r[i] = i;
  }
  uint rg = right;
  uint l_diff;//
  if(MATCH_RATE != 0){
    l_diff = left/(MATCH_RATE*right);
  }else{
    l_diff = 1;
  }
  for (uint i = 0; i < left; i++) {
    lt[i].key = getTupleId();
    if(i%l_diff == 0 && counter < MATCH_RATE*right){
      uint temp = rand()%rg;
      uint temp2 = used_r[temp];
      rg = rg-1;
      used[temp] = used[rg];
      
      lt[i].val = rt[temp2].val;      
      counter++;
    }else{
      uint temp = rand()%selec;
      uint temp2 = used[temp];
      selec = selec-1;
      used[temp] = used[selec];
      lt[i].val = temp2; 
    }
  }
  
  free(used);
  free(used_r);

  shuffle(lt,left);

  res = cuMemHostAlloc((void**)&jt,JT_SIZE * sizeof(RESULT),CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc to LEFT_TUPLE failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
}






/*      memory free           */
void freeTuple(){

  cuMemFreeHost(rt);
  cuMemFreeHost(lt);
  cuMemFreeHost(jt);

}



void join(){

  //uint *count;
  uint jt_size;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUfunction function,c_function,rcp_function,rp_function,sp_function;
  CUmodule module,p_module;
  CUdeviceptr lt_dev, rt_dev, jt_dev, bucket_dev, buckArray_dev ,idxcount_dev;
  CUdeviceptr prt_dev,rL_dev;
  CUdeviceptr ltn_dev, rtn_dev, jt_size_dev;
  CUdeviceptr c_dev;
  unsigned int block_x, grid_x;
  char fname[256];
  const char *path=".";
  struct timeval begin, end;
  struct timeval time_join_s,time_join_f,time_send_s,time_send_f;
  struct timeval time_count_s,time_count_f,time_tsend_s,time_tsend_f,time_isend_s,time_isend_f;
  struct timeval time_jdown_s,time_jdown_f,time_jkernel_s,time_jkernel_f;
  struct timeval time_scan_s,time_scan_f,time_alloc_s,time_alloc_f;
  struct timeval time_hashcreate_s,time_hashcreate_f;
  //double time_cal;




  /******************** GPU init here ************************************************/
  //GPU仕様のために

  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /*********************************************************************************/



  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad() failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(join) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count) failed\n");
    exit(1);
  }


  sprintf(fname, "%s/partitioning.cubin", path);
  res = cuModuleLoad(&p_module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(partitioning) failed\n");
    exit(1);
  }

  res = cuModuleGetFunction(&rcp_function, p_module, "rcount_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(rcount_partitioning) failed\n");
    exit(1);
  }
  res = cuModuleGetFunction(&rp_function, p_module, "rpartitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(rpartitioning) failed\n");
    exit(1);
  }

  res = cuModuleGetFunction(&sp_function, p_module, "countPartition");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(countpartition) failed\n");
    exit(1);
  }


  /*tuple and index init******************************************/  

  createTuple();

  /*
  TUPLE *tlr;
  int lr;
  tlr = lt;
  lt = rt;
  rt = tlr;
  lr = left;
  left = right;
  right = lr;
  */


  gettimeofday(&begin, NULL);
  /****************************************************************/

  /********************************************************************
   *lt,rt,countのメモリを割り当てる。
   *
   */
  gettimeofday(&time_alloc_s, NULL);

  /* lt */
  res = cuMemAlloc(&lt_dev, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    exit(1);
  }
  /* rt */
  res = cuMemAlloc(&rt_dev, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    exit(1);
  }
  /*count */
  res = cuMemAlloc(&c_dev, (left+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }

  gettimeofday(&time_alloc_f, NULL);



  /**********************************************************************************/


  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/

  gettimeofday(&time_send_s, NULL);
  gettimeofday(&time_tsend_s, NULL);

  res = cuMemcpyHtoD(lt_dev, lt, left * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", res);//conv(res));
    exit(1);
  }
  res = cuMemcpyHtoD(rt_dev, rt, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_tsend_f, NULL);
  gettimeofday(&time_send_f, NULL);


  /*
  gettimeofday(&time_cinit_s, NULL);
  
  gettimeofday(&time_cinit_f, NULL);
  */

  /***************************************************************************/

  int p_num = 0;
  int t_num;

  int pt = right*PART_STANDARD;
  //if(right%PART_STANDARD!=0) pt++;

  for(uint i=PARTITION ; i<=pow(PARTITION,4); i*=PARTITION){
    if(i<=pt&&pt<=i*2){
      p_num = i;
    }
  }

  if(p_num==0){
    double temp = right*PART_STANDARD;
    if(temp < 2){
      p_num = 1;
    }else if(floor(log2(temp))==ceil(log2(temp))){
      p_num = (int)temp;
    }else{
      p_num = pow(2,(int)log2(temp) + 1);
    }
  }


  t_num = right/RIGHT_PER_TH;
  if(left%RIGHT_PER_TH != 0){
    t_num++;
  }


  /*hash table create*/
  gettimeofday(&time_hashcreate_s, NULL);

  res = cuMemAlloc(&prt_dev, right * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (prt) failed\n");
    exit(1);
  }

  res = cuMemAlloc(&rL_dev, t_num * PARTITION * sizeof(TUPLE));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (rL) failed\n");
    exit(1);
  }

  t_num = right/RIGHT_PER_TH;
  if(right%RIGHT_PER_TH != 0){
    t_num++;
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);

  /*
  res = cuMemAlloc(&rL_dev, t_num * PARTITION * sizeof(uint));
  if (res != CUDA_SUCCESS){
    printf("cuMemAlloc (rL) failed\n");
    exit(1);
  }
  checkCudaErrors(cudaMemset((void *)rL_dev, 0 , t_num*PARTITION*sizeof(uint)));
  */

  int p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  int p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;

  int p_n=0;
  CUdeviceptr hashtemp;

  for(uint loop=0 ; pow(PARTITION,loop)<p_num ; loop++){

    if(p_num<pow(PARTITION,loop+1)){
      p_n = p_num/pow(PARTITION,loop);
    }else{
      p_n = PARTITION;
    }

    printf("p_grid=%d\tp_block=%d\n",p_grid_x,p_block_x);

    void *count_rpartition_args[]={

      (void *)&rt_dev,
      (void *)&rL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&right,
      (void *)&loop

    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         rcp_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         count_rpartition_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rcount hash) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash count) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }

    /**************************** prefix sum *************************************/
    if(!(presum(&rL_dev,t_num*p_n))){
      printf("lL presum error\n");
      exit(1);
    }
    /********************************************************************/
    void *rpartition_args[]={

      (void *)&rt_dev,
      (void *)&prt_dev,
      (void *)&rL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&right,
      (void *)&loop
    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         rp_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         rpartition_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rhash partition) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash partition) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }

    printf("...loop finish\n");

    hashtemp = rt_dev;
    rt_dev = prt_dev;
    prt_dev = hashtemp;

  }

  p_block_x = 256;
  p_grid_x = right/p_block_x;
  if(right%p_block_x!=0){
    p_block_x++;
  }

  CUdeviceptr rstartPos_dev;
  int rpos_size = MAX_LARGE_ARRAY_SIZE*iDivUp(p_num+1,MAX_LARGE_ARRAY_SIZE);

  res = cuMemAlloc(&rstartPos_dev, rpos_size * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (rstartPos) failed\n");
    exit(1);
  }
  checkCudaErrors(cudaMemset((void *)rstartPos_dev,0,rpos_size*sizeof(uint)));

  void *rspartition_args[]={

    (void *)&rt_dev,
    (void *)&rstartPos_dev,
    (void *)&p_num,
    (void *)&right
  };
  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       sp_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       rspartition_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("cuLaunchKernel(lhash partition) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(lhash partition) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }


  /*
  int r_p_max;
  uint *r_p =  (uint *)calloc(p_num,sizeof(uint));
  res = cuMemcpyDtoH(r_p,rstartPos_dev,p_num * sizeof(uint));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (r_p) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  r_p_max = r_p[0];

  for(uint i = 1; i<p_num ;i++){
    if(r_p[i] > r_p_max){
      r_p_max = r_p[i];
    }
  }
  */

  if(!(presum(&rstartPos_dev,p_num+1))){
    printf("rstartpos presum error\n");
    exit(1);
  }

  res = cuMemFree(prt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (prt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(rL_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rL) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  gettimeofday(&time_hashcreate_f, NULL);


  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/

  gettimeofday(&time_count_s, NULL);


  block_x = left < BLOCK_SIZE_X ? left : BLOCK_SIZE_X;
  grid_x = left / block_x;
  if (left % block_x != 0)
    grid_x++;


  void *count_args[]={
    
    (void *)&lt_dev,
    (void *)&c_dev,
    (void *)&rt_dev,
    (void *)&rstartPos_dev,
    (void *)&p_num,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       count_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS) {
    printf("count cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    exit(1);
  }


  /***************************************************************************************/



  /**************************** prefix sum *************************************/
  gettimeofday(&time_scan_s, NULL);

  if(!(presum(&c_dev,(uint)left+1))){
    printf("count scan error\n");
    exit(1);
  }

  gettimeofday(&time_scan_f, NULL);

  /********************************************************************/

  
  gettimeofday(&time_count_f, NULL);



  /************************************************************************
   join

   jt memory alloc and jt upload
  ************************************************************************/

  gettimeofday(&time_join_s, NULL);

  if(!transport(c_dev,(uint)left+1,&jt_size)){
    printf("transport error.\n");
    exit(1);
  }

  if(jt_size <= 0){
    printf("no tuple is matched.\n");

  }else{
  
    res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (join) failed\n");
      exit(1);
    }

    gettimeofday(&time_jkernel_s, NULL);

    void *kernel_args[]={
      (void *)&rt_dev,
      (void *)&lt_dev,
      (void *)&jt_dev,
      (void *)&rstartPos_dev,
      (void *)&c_dev,
      (void *)&p_num,
      (void *)&left    
    };

    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         function,      // CUfunction f
                         grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         kernel_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("join cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  



    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
      exit(1);
    }  

    gettimeofday(&time_jkernel_f, NULL);

    gettimeofday(&time_join_f, NULL);


    gettimeofday(&time_jdown_s, NULL);

    res = cuMemcpyDtoH(jt, jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
      exit(1);
    }
    gettimeofday(&time_jdown_f, NULL);
    //printf("jt_size = %d\n",jt_size*sizeof(RESULT)/1000);


  }

  /***************************************************************
  free GPU memory
  ***************************************************************/

  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  res = cuMemFree(c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  res = cuMemFree(rstartPos_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rstartPos) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  



  gettimeofday(&end, NULL);


  printf("\n************execution time****************\n\n");
  printf("all time:\n");
  printDiff(begin, end);
  printf("\n");
  printf("gpu memory alloc time:\n");
  printDiff(time_alloc_s,time_alloc_f);
  printf("\n");
  printf("data send time:\n");
  printDiff(time_send_s,time_send_f);
  printf("table data send time:\n");
  printDiff(time_tsend_s,time_tsend_f);
  printf("\n");
  printf("hash table create time:\n");
  printDiff(time_hashcreate_s,time_hashcreate_f);
  printf("\n");
  printf("count time:\n");
  printDiff(time_count_s,time_count_f);
  printf("count scan time:\n");
  printDiff(time_scan_s,time_scan_f);
  printf("\n");
  printf("join time:\n");
  printDiff(time_join_s,time_join_f);
  printf("kernel launch time of join:\n");
  printDiff(time_jkernel_s,time_jkernel_f);
  printf("download time of jt:\n");
  printDiff(time_jdown_s,time_jdown_f);


  printf("%d\n",jt_size);
  

  for(uint i=0;i<3&&i<jt_size;i++){
    printf("join[%d]:left %8d \t:right: %8d\n",i,jt[i].lkey,jt[i].rkey);
    printf("left = %8d\tright = %8d\n",jt[i].lval,jt[i].rval);
    printf("\n");
  }

  /*
  for(int i = 0; i < count[right - 1] ;i++){

    if(jt[i].lval == jt[i].rval && i % 100000==0){

      printf("lid=%d  left=%d\trid=%d  right=%d\n",jt[i].lkey,jt[i].lval,jt[i].rkey,jt[i].rval);
      //printf("left=%d\tright=%d\n",jt[i].lval,jt[i].rval);
    }
  }
  */


  //finish GPU   ****************************************************
  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  

  res = cuModuleUnload(p_module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }  
  
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /****************************************************************************/

  freeTuple();


}


int 
main(int argc,char *argv[])
{


  if(argc>3){
    printf("引数が多い\n");
    return 0;
  }else if(argc<3){
    printf("引数が足りない\n");
    return 0;
  }else{
    left=atoi(argv[1]);
    right=atoi(argv[2]);

    printf("left=%d:right=%d\n",left,right);
  }

  join();

  return 0;
}
