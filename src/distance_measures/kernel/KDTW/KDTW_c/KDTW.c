/*  Wrapping KDTW similarity function with the Python-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/* ==== Square Euclidean Distance as local kernel  ======================
*/
static double sqEuclidean(const double *a, const double *b, int dim){
double x, out=0.0;
for (int k=0; k<dim; k++){
  x=(a[k]-b[k]);
  out+=x*x;
  }
return out;
}

/* ==== Build the local kernel cost matrix ======================
*/
static double **localKernel(double **A, int la, double **B, int lb, int dim, double sigma, double epsilon){
    int l=la;
    if (l<lb)
	l=lb;
    double **LK = (double **)calloc(l, sizeof(double *));
    for (int i = 0; i < l; i++) {
    	LK[i] = (double *)calloc(l, sizeof(double));
        for (int j=0; j<l; j++) {
	    if (i<la && j<lb)
                LK[i][j]=(exp(-sqEuclidean(A[i],B[j], dim)/sigma)+epsilon)/(3*(1+epsilon));
            else if (i<la)
		LK[i][j]=(exp(-sqEuclidean(A[i],B[lb-1], dim)/sigma)+epsilon)/(3*(1+epsilon));
            else 
		LK[i][j]=(exp(-sqEuclidean(A[la-1],B[j], dim)/sigma)+epsilon)/(3*(1+epsilon));
        }
    }
    return LK;
}

/* ==== Evaluate KDTW given the local kernel matrix ======================
*/
double kdtw_lk(int lA, int lB, double **local_kernel){
    int la = lA+1;
    int lb = lB+1;
    int lmx = la;
    int l = lb;
    if (lb > la){
	lmx = lb;
        l = la;
    }
    double **DP = (double **)calloc(la, sizeof(double *));
    double **DP1 = (double **)calloc(la, sizeof(double *));
    for (int i = 0; i < la; i++) {
    	DP[i] = (double *)calloc(lb, sizeof(double));
	DP1[i] = (double *)calloc(lb, sizeof(double));
    }

    DP[0][0] = 1;
    DP1[0][0] = 1;

    for (int i = 1; i<la; i++){
        for (int j=1; j<lb; j++){ 
            DP[i][j] = (DP[i-1][j] + DP[i][j-1] + DP[i-1][j-1]) * local_kernel[i-1][j-1];
            if (i == j)
                DP1[i][j] = DP1[i-1][j-1] * local_kernel[i-1][j-1] + 
			    DP1[i-1][j] * local_kernel[i-1][i-1] + DP1[i][j-1]  *local_kernel[j-1][j-1];
            else
                DP1[i][j] = DP1[i-1][j] * local_kernel[i-1][i-1] + DP1[i][j-1] * local_kernel[j-1][j-1];
    	}
    }
    double ans=DP[la-1][lb-1]+DP1[la-1][lb-1];
    for (int i=0; i<la; i++){
	free(DP[i]);
        free(DP1[i]);
    }
    free(DP);
    free(DP1);

    return ans;
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin, int *L, int *D)  {
	double **c, *a;
	int i,n,m;
	
	n=arrayin->dimensions[0];
	m=arrayin->dimensions[1];
	*L=n;
	*D=m;
	c=ptrvector(n);
	a=(double *) arrayin->data;  /* pointer to arrayin data as double */
	for (i = 0; i < n; i++)  
		c[i] = a + i*m;  

	return c;
}

/* ==== Error tracing ======================
*/
PyObject*  failure(int errid, char* mess){
  printf("%s\n",mess);
  return Py_BuildValue("d", -1.0);
}

/* ==== Python/C bydings for KDTW similarity computation ======================
*/
static PyObject* similarity(PyObject* self, PyObject* args)
{
    double answer=0.0;
    double **A;
    double **B;
    double sigma, epsilon;
    PyArrayObject *seq1, *seq2;
    
    if (!PyArg_ParseTuple(args, "OOdd",  &seq1,  &seq2, &sigma, &epsilon))
        return failure(-1, "TPyArg_ParseTuple error.");

    if (PyArray_DESCR(seq1)->type_num != NPY_DOUBLE)
        return failure(-1, "Type np.float64 expected for p array.");

    if (PyArray_DESCR(seq1)->type_num != NPY_DOUBLE)
        return failure(-1, "Type np.float64 expected for M array.");

    if (PyArray_NDIM(seq1)!=2)
        return failure(-1, "p must be a 2 dimensionnal array.");
    if (PyArray_NDIM(seq2)!=2)
        return failure(-1, "p must be a 2 dimensionnal array.");

    int la, lb, dima, dimb;
    A = pymatrix_to_Carrayptrs(seq1, &la, &dima);
    B = pymatrix_to_Carrayptrs(seq2, &lb, &dimb);

    if (dima != dimb){
        printf("dimensions of time series are not equal! \n");
        return Py_BuildValue("f", -1.0);
        }

   double **LK = localKernel(A, la, B, lb, dima, sigma, epsilon);
   free(A);
   free(B);
   answer = kdtw_lk(la, lb, LK);
   for (int i=0; i<la; i++)
       free(LK[i]);
   free(LK);
   /*  construct the output, from c double to python float */
   return Py_BuildValue("d", answer);
}

/*  define functions in module */
static PyMethodDef similarityMethods[] =
{
     {"similarity", similarity, METH_VARARGS, "evaluate kdtw with double 2D-arrays args"},
     {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "KDTW", "Some documentation",
    -1,
    similarityMethods
};

PyMODINIT_FUNC
PyInit_KDTW(void)
{
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initKDTW(void)
{
    (void) Py_InitModule("KDTW", similarityMethods);
}

#endif

