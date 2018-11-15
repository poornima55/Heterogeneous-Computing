void reduce(                                          
   __local  float*,                          
   __global float*);



__kernel void pi_compute(int num_per_item, __local float* local_result, 
 __global float* global_result) {
   
   int local_id = get_local_id(0);
   printf("local_id is %d\n", local_id);
   float  accum = 0.0f;  
   int id = get_global_id(0);
   printf("Global id is %d\n",id);
   char sign = 0;
   /* Make sure previous processing has completed */
   
   /* barrier(CLK_LOCAL_MEM_FENCE);*/

 
   /* Each work item will compute sum over 4 terms */
   for(int i=0; i<num_per_item; i++) {
          
          if(sign ==0)
          {    
          	accum+= (1.0f/((2*id+i)+1.0f));
                sign = 1;
           }
          else
           {    
          	accum+= (-1.0f/((2*id+i)+1.0f));
                sign = 0;
           }
            
            
        }
   local_result[local_id] = accum;    //accumulated result of that particular work item
   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);
   
   reduce(local_result, global_result); //after each workitem finishes, check it if is 1st work item, if so add it to the global result array 
}


void reduce(                                          
   __local  float*    local_result,                          
   __global float*    global_result)                        
{                                                          
   int num_wrk_items  = get_local_size(0); 
   printf("Num of work items=%d\n",num_wrk_items);                
   int local_id       = get_local_id(0);                   
   int group_id       = get_group_id(0);                   
   
   float sum;                              
   int i;                                      
   
   if (local_id == 0) {                      
      sum = 0.0f;                            
   
      for (i=0; i<num_wrk_items-8; i++) {        
          sum += local_result[i];             
      }                                     
   
      *global_result= sum;         
   }
}
