/* widthA=heightB for valid matrix multiplication */
/* matrices dimensions should be same for matrix addition*/
__kernel void simpleMultiplyAdd(
    __global float *outputD,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC,
    int heightC,
    __global float *inputA,
    __global float *inputB,
    __global float *inputC)
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);

    float sum = 0.0f;

    /* calculate result of one element of Matrix C */
    for (int i=0; i<widthA; i++) {
        sum += inputA[row*widthA + i] * inputB[i*widthB + col];
    }

    outputD[row*widthB + col] = sum + inputC[row*widthB+col];
}
