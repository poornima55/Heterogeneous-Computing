void reduce(                                          
   __local  float*,                          
   __global float*);



__kernel void pi_compute(__global int* id,int num_per_item, __local float* local_result, 
 __global float* global_result) {
   

   int local_id = get_local_id(0);
   float  accum = 0.0f;  
   char sign = 0;
   /* Make sure previous processing has completed */
   
   /* barrier(CLK_LOCAL_MEM_FENCE);*/

 
   /* Each work item will compute sum over 4 terms */
   for(int i=0; i<num_per_item; i++) {
          
          if(sign ==0)
          {    
          	accum+= (1.0f/((2*(*id))+1.0f));
                sign = 1;
           }
          else
           {    
          	accum+= (-1.0f/((2*(*id))+1.0f));
                sign = 0;
           }
           
           (*id)=(*id)+1;
            
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
    int local_id = get_local_id(0);
                                                     
   
   if (local_id == 0) { 
      /* assign it to the global result*/ 
      int i;
      for(i=0;i<num_wrk_items; i++)                   
      global_result[i] = local_result[i];
   }

}
