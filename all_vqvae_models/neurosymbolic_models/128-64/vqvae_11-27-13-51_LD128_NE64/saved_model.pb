??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMin

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.12v2.10.0-76-gfdfc646704c8??
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:  *
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameconv2d_transpose_2/kernel
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:?*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:@?*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
}
embeddings_vqvaeVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*!
shared_nameembeddings_vqvae
v
$embeddings_vqvae/Read/ReadVariableOpReadVariableOpembeddings_vqvae*
_output_shapes
:	?@*
dtype0
?
serving_default_input_4Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasembeddings_vqvaeconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_126652

NoOpNoOp
?b
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?a
value?aB?a B?a
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
?
 layer-0
!layer_with_weights-0
!layer-1
"layer_with_weights-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer_with_weights-4
%layer-5
&layer_with_weights-5
&layer-6
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
?
-0
.1
/2
03
14
25
36
47
58
69
10
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22*
?
-0
.1
/2
03
14
25
36
47
58
69
10
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22*
* 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 

Pserving_default* 
* 
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

-kernel
.bias
 W_jit_compiled_convolution_op*
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

/kernel
0bias
 ^_jit_compiled_convolution_op*
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

1kernel
2bias
 e_jit_compiled_convolution_op*
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

3kernel
4bias
 l_jit_compiled_convolution_op*
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

5kernel
6bias
 s_jit_compiled_convolution_op*
J
-0
.1
/2
03
14
25
36
47
58
69*
J
-0
.1
/2
03
14
25
36
47
58
69*
* 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ytrace_0
ztrace_1
{trace_2
|trace_3* 
7
}trace_0
~trace_1
trace_2
?trace_3* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
d^
VARIABLE_VALUEembeddings_vqvae:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

7kernel
8bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

9kernel
:bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

;kernel
<bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

=kernel
>bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

?kernel
@bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Akernel
Bbias
!?_jit_compiled_convolution_op*
Z
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11*
Z
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_4/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

/0
01*

/0
01*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

10
21*

10
21*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

30
41*

30
41*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

50
61*

50
61*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

70
81*

70
81*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

90
:1*

90
:1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

=0
>1*

=0
>1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

?0
@1*

?0
@1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

A0
B1*

A0
B1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
5
 0
!1
"2
#3
$4
%5
&6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$embeddings_vqvae/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOpConst*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_128101
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembeddings_vqvaeconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_128180??
?!
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?.
?
C__inference_encoder_layer_call_and_return_conditional_losses_127252

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@B
'conv2d_4_conv2d_readvariableop_resource:@?7
(conv2d_4_biasadd_readvariableop_resource:	?
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?q
IdentityIdentityconv2d_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_127662

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_125276w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_125479

inputs'
conv2d_125453: 
conv2d_125455: )
conv2d_1_125458: @
conv2d_1_125460:@)
conv2d_2_125463:@@
conv2d_2_125465:@)
conv2d_3_125468:@@
conv2d_3_125470:@*
conv2d_4_125473:@?
conv2d_4_125475:	?
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_125453conv2d_125455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_125276?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_125458conv2d_1_125460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_125463conv2d_2_125465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_125468conv2d_3_125470*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_125473conv2d_4_125475*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_2_layer_call_fn_127702

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126387

inputs(
encoder_126335: 
encoder_126337: (
encoder_126339: @
encoder_126341:@(
encoder_126343:@@
encoder_126345:@(
encoder_126347:@@
encoder_126349:@)
encoder_126351:@?
encoder_126353:	?*
vector_quantizer_126356:	?@)
decoder_126360:@?
decoder_126362:@(
decoder_126364:@@
decoder_126366:@(
decoder_126368:@@
decoder_126370:@(
decoder_126372: @
decoder_126374: (
decoder_126376:  
decoder_126378: (
decoder_126380: 
decoder_126382:
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?(vector_quantizer/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_126335encoder_126337encoder_126339encoder_126341encoder_126343encoder_126345encoder_126347encoder_126349encoder_126351encoder_126353*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125479?
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_126356*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?????????@@?: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187?
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_126360decoder_126362decoder_126364decoder_126366decoder_126368decoder_126370decoder_126372decoder_126374decoder_126376decoder_126378decoder_126380decoder_126382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125984
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
'__inference_vq_vae_layer_call_fn_126756

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
	unknown_9:	?@%

unknown_10:@?

unknown_11:@$

unknown_12:@@

unknown_13:@$

unknown_14:@@

unknown_15:@$

unknown_16: @

unknown_17: $

unknown_18:  

unknown_19: $

unknown_20: 

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_vq_vae_layer_call_and_return_conditional_losses_126387w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_127407

inputs"
unknown:@?
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5: @
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125984w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?&
?
C__inference_decoder_layer_call_and_return_conditional_losses_125892

inputs2
conv2d_transpose_125861:@?%
conv2d_transpose_125863:@3
conv2d_transpose_1_125866:@@'
conv2d_transpose_1_125868:@3
conv2d_transpose_2_125871:@@'
conv2d_transpose_2_125873:@3
conv2d_transpose_3_125876: @'
conv2d_transpose_3_125878: 3
conv2d_transpose_4_125881:  '
conv2d_transpose_4_125883: 3
conv2d_transpose_5_125886: '
conv2d_transpose_5_125888:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_125861conv2d_transpose_125863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_125866conv2d_transpose_1_125868*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_125871conv2d_transpose_2_125873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_125876conv2d_transpose_3_125878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_125881conv2d_transpose_4_125883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_125886conv2d_transpose_5_125888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847?
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?&
?
C__inference_decoder_layer_call_and_return_conditional_losses_125984

inputs2
conv2d_transpose_125953:@?%
conv2d_transpose_125955:@3
conv2d_transpose_1_125958:@@'
conv2d_transpose_1_125960:@3
conv2d_transpose_2_125963:@@'
conv2d_transpose_2_125965:@3
conv2d_transpose_3_125968: @'
conv2d_transpose_3_125970: 3
conv2d_transpose_4_125973:  '
conv2d_transpose_4_125975: 3
conv2d_transpose_5_125978: '
conv2d_transpose_5_125980:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_125953conv2d_transpose_125955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_125958conv2d_transpose_1_125960*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_125963conv2d_transpose_2_125965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_125968conv2d_transpose_3_125970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_125973conv2d_transpose_4_125975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_125978conv2d_transpose_5_125980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847?
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
1__inference_vector_quantizer_layer_call_fn_127298
x
unknown:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?????????@@?: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????@@?

_user_specified_namex
?
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126544
input_4(
encoder_126492: 
encoder_126494: (
encoder_126496: @
encoder_126498:@(
encoder_126500:@@
encoder_126502:@(
encoder_126504:@@
encoder_126506:@)
encoder_126508:@?
encoder_126510:	?*
vector_quantizer_126513:	?@)
decoder_126517:@?
decoder_126519:@(
decoder_126521:@@
decoder_126523:@(
decoder_126525:@@
decoder_126527:@(
decoder_126529: @
decoder_126531: (
decoder_126533:  
decoder_126535: (
decoder_126537: 
decoder_126539:
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?(vector_quantizer/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_4encoder_126492encoder_126494encoder_126496encoder_126498encoder_126500encoder_126502encoder_126504encoder_126506encoder_126508encoder_126510*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125350?
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_126513*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?????????@@?: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187?
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_126517decoder_126519decoder_126521decoder_126523decoder_126525decoder_126527decoder_126529decoder_126531decoder_126533decoder_126535decoder_126537decoder_126539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125892
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?
?
)__inference_conv2d_3_layer_call_fn_127722

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_4_layer_call_fn_127933

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
'__inference_vq_vae_layer_call_fn_126704

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
	unknown_9:	?@%

unknown_10:@?

unknown_11:@$

unknown_12:@@

unknown_13:@$

unknown_14:@@

unknown_15:@$

unknown_16: @

unknown_17: $

unknown_18:  

unknown_19: $

unknown_20: 

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_vq_vae_layer_call_and_return_conditional_losses_126219w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126219

inputs(
encoder_126115: 
encoder_126117: (
encoder_126119: @
encoder_126121:@(
encoder_126123:@@
encoder_126125:@(
encoder_126127:@@
encoder_126129:@)
encoder_126131:@?
encoder_126133:	?*
vector_quantizer_126188:	?@)
decoder_126192:@?
decoder_126194:@(
decoder_126196:@@
decoder_126198:@(
decoder_126200:@@
decoder_126202:@(
decoder_126204: @
decoder_126206: (
decoder_126208:  
decoder_126210: (
decoder_126212: 
decoder_126214:
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?(vector_quantizer/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_126115encoder_126117encoder_126119encoder_126121encoder_126123encoder_126125encoder_126127encoder_126129encoder_126131encoder_126133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125350?
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_126188*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?????????@@?: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187?
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_126192decoder_126194decoder_126196decoder_126198decoder_126200decoder_126202decoder_126204decoder_126206decoder_126208decoder_126210decoder_126212decoder_126214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125892
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_127378

inputs"
unknown:@?
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5: @
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125892w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_127838

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
C__inference_decoder_layer_call_and_return_conditional_losses_127530

inputsT
9conv2d_transpose_conv2d_transpose_readvariableop_resource:@?>
0conv2d_transpose_biasadd_readvariableop_resource:@U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_1_biasadd_readvariableop_resource:@U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_2_biasadd_readvariableop_resource:@U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_3_biasadd_readvariableop_resource: U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:  @
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@~
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_126040
input_3"
unknown:@?
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5: @
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125984w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????@@?
!
_user_specified_name	input_3
?)
?
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_127349
x1
matmul_readvariableop_resource:	?@
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   `
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:??????????u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0s
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
powPowReshape:output:0pow/y:output:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(n
ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
pow_1PowReadVariableOp:value:0pow_1/y:output:0*
T0*
_output_shapes
:	?@Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ^
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:@\
addAddV2Sum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
mulMulmul/x:output:0MatMul:product:0*
T0*'
_output_shapes
:?????????@N
subSubadd:z:0mul:z:0*
T0*'
_output_shapes
:?????????@R
ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :b
ArgMinArgMinsub:z:0ArgMin/dimension:output:0*
T0*#
_output_shapes
:?????????U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :@?
one_hotOneHotArgMin:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*'
_output_shapes
:?????????@w
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
MatMul_1MatMulone_hot:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(s
	Reshape_1ReshapeMatMul_1:product:0Shape:output:0*
T0*0
_output_shapes
:?????????@@?k
StopGradientStopGradientReshape_1:output:0*
T0*0
_output_shapes
:?????????@@?a
sub_1SubStopGradient:output:0x*
T0*0
_output_shapes
:?????????@@?L
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
pow_2Pow	sub_1:z:0pow_2/y:output:0*
T0*0
_output_shapes
:?????????@@?^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
MeanMean	pow_2:z:0Const:output:0*
T0*
_output_shapes
: \
StopGradient_1StopGradientx*
T0*0
_output_shapes
:?????????@@?t
sub_2SubReshape_1:output:0StopGradient_1:output:0*
T0*0
_output_shapes
:?????????@@?L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
pow_3Pow	sub_2:z:0pow_3/y:output:0*
T0*0
_output_shapes
:?????????@@?`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             L
Mean_1Mean	pow_3:z:0Const_1:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>N
mul_1Mulmul_1/x:output:0Mean:output:0*
T0*
_output_shapes
: K
add_1AddV2	mul_1:z:0Mean_1:output:0*
T0*
_output_shapes
: ^
sub_3SubReshape_1:output:0x*
T0*0
_output_shapes
:?????????@@?d
StopGradient_2StopGradient	sub_3:z:0*
T0*0
_output_shapes
:?????????@@?e
add_2AddV2xStopGradient_2:output:0*
T0*0
_output_shapes
:?????????@@?a
IdentityIdentity	add_2:z:0^NoOp*
T0*0
_output_shapes
:?????????@@?I

Identity_1Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:S O
0
_output_shapes
:?????????@@?

_user_specified_namex
?

?
(__inference_encoder_layer_call_fn_127214

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125479x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_125919
input_3"
unknown:@?
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5: @
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125892w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????@@?
!
_user_specified_name	input_3
?

?
(__inference_encoder_layer_call_fn_127189

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125350x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126599
input_4(
encoder_126547: 
encoder_126549: (
encoder_126551: @
encoder_126553:@(
encoder_126555:@@
encoder_126557:@(
encoder_126559:@@
encoder_126561:@)
encoder_126563:@?
encoder_126565:	?*
vector_quantizer_126568:	?@)
decoder_126572:@?
decoder_126574:@(
decoder_126576:@@
decoder_126578:@(
decoder_126580:@@
decoder_126582:@(
decoder_126584: @
decoder_126586: (
decoder_126588:  
decoder_126590: (
decoder_126592: 
decoder_126594:
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?(vector_quantizer/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_4encoder_126547encoder_126549encoder_126551encoder_126553encoder_126555encoder_126557encoder_126559encoder_126561encoder_126563encoder_126565*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125479?
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_126568*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?????????@@?: *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187?
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_126572decoder_126574decoder_126576decoder_126578decoder_126580decoder_126582decoder_126584decoder_126586decoder_126588decoder_126590decoder_126592decoder_126594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_125984
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?4
?

__inference__traced_save_128101
file_prefix/
+savev2_embeddings_vqvae_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_embeddings_vqvae_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@: : : @:@:@@:@:@@:@:@?:?:@?:@:@@:@:@@:@: @: :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 	

_output_shapes
:@:-
)
'
_output_shapes
:@?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_127693

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_127673

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_127881

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_126652
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
	unknown_9:	?@%

unknown_10:@?

unknown_11:@$

unknown_12:@@

unknown_13:@$

unknown_14:@@

unknown_15:@$

unknown_16: @

unknown_17: $

unknown_18:  

unknown_19: $

unknown_20: 

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_125258w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_127733

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?!
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_127795

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?&
?
C__inference_decoder_layer_call_and_return_conditional_losses_126108
input_32
conv2d_transpose_126077:@?%
conv2d_transpose_126079:@3
conv2d_transpose_1_126082:@@'
conv2d_transpose_1_126084:@3
conv2d_transpose_2_126087:@@'
conv2d_transpose_2_126089:@3
conv2d_transpose_3_126092: @'
conv2d_transpose_3_126094: 3
conv2d_transpose_4_126097:  '
conv2d_transpose_4_126099: 3
conv2d_transpose_5_126102: '
conv2d_transpose_5_126104:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_transpose_126077conv2d_transpose_126079*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_126082conv2d_transpose_1_126084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_126087conv2d_transpose_2_126089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_126092conv2d_transpose_3_126094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_126097conv2d_transpose_4_126099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_126102conv2d_transpose_5_126104*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847?
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????@@?
!
_user_specified_name	input_3
?
?
3__inference_conv2d_transpose_5_layer_call_fn_127976

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?)
?
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_126187
x1
matmul_readvariableop_resource:	?@
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   `
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:??????????u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0s
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
powPowReshape:output:0pow/y:output:0*
T0*(
_output_shapes
:??????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(n
ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
pow_1PowReadVariableOp:value:0pow_1/y:output:0*
T0*
_output_shapes
:	?@Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ^
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:@\
addAddV2Sum:output:0Sum_1:output:0*
T0*'
_output_shapes
:?????????@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
mulMulmul/x:output:0MatMul:product:0*
T0*'
_output_shapes
:?????????@N
subSubadd:z:0mul:z:0*
T0*'
_output_shapes
:?????????@R
ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :b
ArgMinArgMinsub:z:0ArgMin/dimension:output:0*
T0*#
_output_shapes
:?????????U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :@?
one_hotOneHotArgMin:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*'
_output_shapes
:?????????@w
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
MatMul_1MatMulone_hot:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(s
	Reshape_1ReshapeMatMul_1:product:0Shape:output:0*
T0*0
_output_shapes
:?????????@@?k
StopGradientStopGradientReshape_1:output:0*
T0*0
_output_shapes
:?????????@@?a
sub_1SubStopGradient:output:0x*
T0*0
_output_shapes
:?????????@@?L
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
pow_2Pow	sub_1:z:0pow_2/y:output:0*
T0*0
_output_shapes
:?????????@@?^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
MeanMean	pow_2:z:0Const:output:0*
T0*
_output_shapes
: \
StopGradient_1StopGradientx*
T0*0
_output_shapes
:?????????@@?t
sub_2SubReshape_1:output:0StopGradient_1:output:0*
T0*0
_output_shapes
:?????????@@?L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
pow_3Pow	sub_2:z:0pow_3/y:output:0*
T0*0
_output_shapes
:?????????@@?`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             L
Mean_1Mean	pow_3:z:0Const_1:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>N
mul_1Mulmul_1/x:output:0Mean:output:0*
T0*
_output_shapes
: K
add_1AddV2	mul_1:z:0Mean_1:output:0*
T0*
_output_shapes
: ^
sub_3SubReshape_1:output:0x*
T0*0
_output_shapes
:?????????@@?d
StopGradient_2StopGradient	sub_3:z:0*
T0*0
_output_shapes
:?????????@@?e
add_2AddV2xStopGradient_2:output:0*
T0*0
_output_shapes
:?????????@@?a
IdentityIdentity	add_2:z:0^NoOp*
T0*0
_output_shapes
:?????????@@?I

Identity_1Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:S O
0
_output_shapes
:?????????@@?

_user_specified_namex
??
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_127164

inputsG
-encoder_conv2d_conv2d_readvariableop_resource: <
.encoder_conv2d_biasadd_readvariableop_resource: I
/encoder_conv2d_1_conv2d_readvariableop_resource: @>
0encoder_conv2d_1_biasadd_readvariableop_resource:@I
/encoder_conv2d_2_conv2d_readvariableop_resource:@@>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@@>
0encoder_conv2d_3_biasadd_readvariableop_resource:@J
/encoder_conv2d_4_conv2d_readvariableop_resource:@??
0encoder_conv2d_4_biasadd_readvariableop_resource:	?B
/vector_quantizer_matmul_readvariableop_resource:	?@\
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:@?F
8decoder_conv2d_transpose_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:  H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity

identity_1??/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?'encoder/conv2d_2/BiasAdd/ReadVariableOp?&encoder/conv2d_2/Conv2D/ReadVariableOp?'encoder/conv2d_3/BiasAdd/ReadVariableOp?&encoder/conv2d_3/Conv2D/ReadVariableOp?'encoder/conv2d_4/BiasAdd/ReadVariableOp?&encoder/conv2d_4/Conv2D/ReadVariableOp?&vector_quantizer/MatMul/ReadVariableOp?(vector_quantizer/MatMul_1/ReadVariableOp?vector_quantizer/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
encoder/conv2d_4/Conv2DConv2D#encoder/conv2d_3/Relu:activations:0.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?g
vector_quantizer/ShapeShape!encoder/conv2d_4/BiasAdd:output:0*
T0*
_output_shapes
:o
vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
vector_quantizer/ReshapeReshape!encoder/conv2d_4/BiasAdd:output:0'vector_quantizer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
&vector_quantizer/MatMul/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
vector_quantizer/MatMulMatMul!vector_quantizer/Reshape:output:0.vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/powPow!vector_quantizer/Reshape:output:0vector_quantizer/pow/y:output:0*
T0*(
_output_shapes
:??????????h
&vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
vector_quantizer/SumSumvector_quantizer/pow:z:0/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(?
vector_quantizer/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0]
vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_1Pow'vector_quantizer/ReadVariableOp:value:0!vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	?@j
(vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
vector_quantizer/Sum_1Sumvector_quantizer/pow_1:z:01vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:@?
vector_quantizer/addAddV2vector_quantizer/Sum:output:0vector_quantizer/Sum_1:output:0*
T0*'
_output_shapes
:?????????@[
vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/mulMulvector_quantizer/mul/x:output:0!vector_quantizer/MatMul:product:0*
T0*'
_output_shapes
:?????????@?
vector_quantizer/subSubvector_quantizer/add:z:0vector_quantizer/mul:z:0*
T0*'
_output_shapes
:?????????@c
!vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :?
vector_quantizer/ArgMinArgMinvector_quantizer/sub:z:0*vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:?????????f
!vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??g
"vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    `
vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :@?
vector_quantizer/one_hotOneHot vector_quantizer/ArgMin:output:0'vector_quantizer/one_hot/depth:output:0*vector_quantizer/one_hot/on_value:output:0+vector_quantizer/one_hot/off_value:output:0*
T0*'
_output_shapes
:?????????@?
(vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
vector_quantizer/MatMul_1MatMul!vector_quantizer/one_hot:output:00vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(?
vector_quantizer/Reshape_1Reshape#vector_quantizer/MatMul_1:product:0vector_quantizer/Shape:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/StopGradientStopGradient#vector_quantizer/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/sub_1Sub&vector_quantizer/StopGradient:output:0!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?]
vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_2Powvector_quantizer/sub_1:z:0!vector_quantizer/pow_2/y:output:0*
T0*0
_output_shapes
:?????????@@?o
vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             {
vector_quantizer/MeanMeanvector_quantizer/pow_2:z:0vector_quantizer/Const:output:0*
T0*
_output_shapes
: ?
vector_quantizer/StopGradient_1StopGradient!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/sub_2Sub#vector_quantizer/Reshape_1:output:0(vector_quantizer/StopGradient_1:output:0*
T0*0
_output_shapes
:?????????@@?]
vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_3Powvector_quantizer/sub_2:z:0!vector_quantizer/pow_3/y:output:0*
T0*0
_output_shapes
:?????????@@?q
vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
vector_quantizer/Mean_1Meanvector_quantizer/pow_3:z:0!vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: ]
vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
vector_quantizer/mul_1Mul!vector_quantizer/mul_1/x:output:0vector_quantizer/Mean:output:0*
T0*
_output_shapes
: ~
vector_quantizer/add_1AddV2vector_quantizer/mul_1:z:0 vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: ?
vector_quantizer/sub_3Sub#vector_quantizer/Reshape_1:output:0!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/StopGradient_2StopGradientvector_quantizer/sub_3:z:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/add_2AddV2!encoder/conv2d_4/BiasAdd:output:0(vector_quantizer/StopGradient_2:output:0*
T0*0
_output_shapes
:?????????@@?h
decoder/conv2d_transpose/ShapeShapevector_quantizer/add_2:z:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@{
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@}
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose_2/ReluRelu+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@}
 decoder/conv2d_transpose_3/ShapeShape-decoder/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ }
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@Z

Identity_1Identityvector_quantizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?	
NoOpNoOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp'^vector_quantizer/MatMul/ReadVariableOp)^vector_quantizer/MatMul_1/ReadVariableOp ^vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2P
&vector_quantizer/MatMul/ReadVariableOp&vector_quantizer/MatMul/ReadVariableOp2T
(vector_quantizer/MatMul_1/ReadVariableOp(vector_quantizer/MatMul_1/ReadVariableOp2B
vector_quantizer/ReadVariableOpvector_quantizer/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126960

inputsG
-encoder_conv2d_conv2d_readvariableop_resource: <
.encoder_conv2d_biasadd_readvariableop_resource: I
/encoder_conv2d_1_conv2d_readvariableop_resource: @>
0encoder_conv2d_1_biasadd_readvariableop_resource:@I
/encoder_conv2d_2_conv2d_readvariableop_resource:@@>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@@>
0encoder_conv2d_3_biasadd_readvariableop_resource:@J
/encoder_conv2d_4_conv2d_readvariableop_resource:@??
0encoder_conv2d_4_biasadd_readvariableop_resource:	?B
/vector_quantizer_matmul_readvariableop_resource:	?@\
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:@?F
8decoder_conv2d_transpose_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@@H
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:  H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity

identity_1??/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?'encoder/conv2d_2/BiasAdd/ReadVariableOp?&encoder/conv2d_2/Conv2D/ReadVariableOp?'encoder/conv2d_3/BiasAdd/ReadVariableOp?&encoder/conv2d_3/Conv2D/ReadVariableOp?'encoder/conv2d_4/BiasAdd/ReadVariableOp?&encoder/conv2d_4/Conv2D/ReadVariableOp?&vector_quantizer/MatMul/ReadVariableOp?(vector_quantizer/MatMul_1/ReadVariableOp?vector_quantizer/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
encoder/conv2d_4/Conv2DConv2D#encoder/conv2d_3/Relu:activations:0.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?g
vector_quantizer/ShapeShape!encoder/conv2d_4/BiasAdd:output:0*
T0*
_output_shapes
:o
vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
vector_quantizer/ReshapeReshape!encoder/conv2d_4/BiasAdd:output:0'vector_quantizer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
&vector_quantizer/MatMul/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
vector_quantizer/MatMulMatMul!vector_quantizer/Reshape:output:0.vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/powPow!vector_quantizer/Reshape:output:0vector_quantizer/pow/y:output:0*
T0*(
_output_shapes
:??????????h
&vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
vector_quantizer/SumSumvector_quantizer/pow:z:0/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(?
vector_quantizer/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0]
vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_1Pow'vector_quantizer/ReadVariableOp:value:0!vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	?@j
(vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
vector_quantizer/Sum_1Sumvector_quantizer/pow_1:z:01vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:@?
vector_quantizer/addAddV2vector_quantizer/Sum:output:0vector_quantizer/Sum_1:output:0*
T0*'
_output_shapes
:?????????@[
vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/mulMulvector_quantizer/mul/x:output:0!vector_quantizer/MatMul:product:0*
T0*'
_output_shapes
:?????????@?
vector_quantizer/subSubvector_quantizer/add:z:0vector_quantizer/mul:z:0*
T0*'
_output_shapes
:?????????@c
!vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :?
vector_quantizer/ArgMinArgMinvector_quantizer/sub:z:0*vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:?????????f
!vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??g
"vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    `
vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :@?
vector_quantizer/one_hotOneHot vector_quantizer/ArgMin:output:0'vector_quantizer/one_hot/depth:output:0*vector_quantizer/one_hot/on_value:output:0+vector_quantizer/one_hot/off_value:output:0*
T0*'
_output_shapes
:?????????@?
(vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
vector_quantizer/MatMul_1MatMul!vector_quantizer/one_hot:output:00vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(?
vector_quantizer/Reshape_1Reshape#vector_quantizer/MatMul_1:product:0vector_quantizer/Shape:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/StopGradientStopGradient#vector_quantizer/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/sub_1Sub&vector_quantizer/StopGradient:output:0!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?]
vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_2Powvector_quantizer/sub_1:z:0!vector_quantizer/pow_2/y:output:0*
T0*0
_output_shapes
:?????????@@?o
vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             {
vector_quantizer/MeanMeanvector_quantizer/pow_2:z:0vector_quantizer/Const:output:0*
T0*
_output_shapes
: ?
vector_quantizer/StopGradient_1StopGradient!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/sub_2Sub#vector_quantizer/Reshape_1:output:0(vector_quantizer/StopGradient_1:output:0*
T0*0
_output_shapes
:?????????@@?]
vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vector_quantizer/pow_3Powvector_quantizer/sub_2:z:0!vector_quantizer/pow_3/y:output:0*
T0*0
_output_shapes
:?????????@@?q
vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
vector_quantizer/Mean_1Meanvector_quantizer/pow_3:z:0!vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: ]
vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
vector_quantizer/mul_1Mul!vector_quantizer/mul_1/x:output:0vector_quantizer/Mean:output:0*
T0*
_output_shapes
: ~
vector_quantizer/add_1AddV2vector_quantizer/mul_1:z:0 vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: ?
vector_quantizer/sub_3Sub#vector_quantizer/Reshape_1:output:0!encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/StopGradient_2StopGradientvector_quantizer/sub_3:z:0*
T0*0
_output_shapes
:?????????@@??
vector_quantizer/add_2AddV2!encoder/conv2d_4/BiasAdd:output:0(vector_quantizer/StopGradient_2:output:0*
T0*0
_output_shapes
:?????????@@?h
decoder/conv2d_transpose/ShapeShapevector_quantizer/add_2:z:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@{
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@}
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
decoder/conv2d_transpose_2/ReluRelu+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@}
 decoder/conv2d_transpose_3/ShapeShape-decoder/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ }
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@Z

Identity_1Identityvector_quantizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?	
NoOpNoOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp'^vector_quantizer/MatMul/ReadVariableOp)^vector_quantizer/MatMul_1/ReadVariableOp ^vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2P
&vector_quantizer/MatMul/ReadVariableOp&vector_quantizer/MatMul/ReadVariableOp2T
(vector_quantizer/MatMul_1/ReadVariableOp(vector_quantizer/MatMul_1/ReadVariableOp2B
vector_quantizer/ReadVariableOpvector_quantizer/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_2_layer_call_fn_127847

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_layer_call_fn_127761

inputs"
unknown:@?
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_125276

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_127924

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_128009

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
'__inference_vq_vae_layer_call_fn_126489
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
	unknown_9:	?@%

unknown_10:@?

unknown_11:@$

unknown_12:@@

unknown_13:@$

unknown_14:@@

unknown_15:@$

unknown_16: @

unknown_17: $

unknown_18:  

unknown_19: $

unknown_20: 

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_vq_vae_layer_call_and_return_conditional_losses_126387w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?
?
'__inference_vq_vae_layer_call_fn_126269
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
	unknown_9:	?@%

unknown_10:@?

unknown_11:@$

unknown_12:@@

unknown_13:@$

unknown_14:@@

unknown_15:@$

unknown_16: @

unknown_17: $

unknown_18:  

unknown_19: $

unknown_20: 

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@@: *9
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_vq_vae_layer_call_and_return_conditional_losses_126219w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_125585
input_1'
conv2d_125559: 
conv2d_125561: )
conv2d_1_125564: @
conv2d_1_125566:@)
conv2d_2_125569:@@
conv2d_2_125571:@)
conv2d_3_125574:@@
conv2d_3_125576:@*
conv2d_4_125579:@?
conv2d_4_125581:	?
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_125559conv2d_125561*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_125276?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_125564conv2d_1_125566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_125569conv2d_2_125571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_125574conv2d_3_125576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_125579conv2d_4_125581*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
? 
?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_127682

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_125556
input_1'
conv2d_125530: 
conv2d_125532: )
conv2d_1_125535: @
conv2d_1_125537:@)
conv2d_2_125540:@@
conv2d_2_125542:@)
conv2d_3_125545:@@
conv2d_3_125547:@*
conv2d_4_125550:@?
conv2d_4_125552:	?
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_125530conv2d_125532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_125276?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_125535conv2d_1_125537*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_125540conv2d_2_125542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_125545conv2d_3_125547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_125550conv2d_4_125552*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?.
?
C__inference_encoder_layer_call_and_return_conditional_losses_127290

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@B
'conv2d_4_conv2d_readvariableop_resource:@?7
(conv2d_4_biasadd_readvariableop_resource:	?
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?q
IdentityIdentityconv2d_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
C__inference_decoder_layer_call_and_return_conditional_losses_127653

inputsT
9conv2d_transpose_conv2d_transpose_readvariableop_resource:@?>
0conv2d_transpose_biasadd_readvariableop_resource:@U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_1_biasadd_readvariableop_resource:@U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@@@
2conv2d_transpose_2_biasadd_readvariableop_resource:@U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_3_biasadd_readvariableop_resource: U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:  @
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@~
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
(__inference_encoder_layer_call_fn_125373
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125350x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
??
?
!__inference__wrapped_model_125258
input_4N
4vq_vae_encoder_conv2d_conv2d_readvariableop_resource: C
5vq_vae_encoder_conv2d_biasadd_readvariableop_resource: P
6vq_vae_encoder_conv2d_1_conv2d_readvariableop_resource: @E
7vq_vae_encoder_conv2d_1_biasadd_readvariableop_resource:@P
6vq_vae_encoder_conv2d_2_conv2d_readvariableop_resource:@@E
7vq_vae_encoder_conv2d_2_biasadd_readvariableop_resource:@P
6vq_vae_encoder_conv2d_3_conv2d_readvariableop_resource:@@E
7vq_vae_encoder_conv2d_3_biasadd_readvariableop_resource:@Q
6vq_vae_encoder_conv2d_4_conv2d_readvariableop_resource:@?F
7vq_vae_encoder_conv2d_4_biasadd_readvariableop_resource:	?I
6vq_vae_vector_quantizer_matmul_readvariableop_resource:	?@c
Hvq_vae_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:@?M
?vq_vae_decoder_conv2d_transpose_biasadd_readvariableop_resource:@d
Jvq_vae_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@@O
Avq_vae_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:@d
Jvq_vae_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@@O
Avq_vae_decoder_conv2d_transpose_2_biasadd_readvariableop_resource:@d
Jvq_vae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @O
Avq_vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource: d
Jvq_vae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:  O
Avq_vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource: d
Jvq_vae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: O
Avq_vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity??6vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp??vq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?8vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?Avq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?8vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?Avq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp?Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp?Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp?Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?,vq_vae/encoder/conv2d/BiasAdd/ReadVariableOp?+vq_vae/encoder/conv2d/Conv2D/ReadVariableOp?.vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOp?-vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOp?.vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOp?-vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOp?.vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOp?-vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOp?.vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOp?-vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOp?-vq_vae/vector_quantizer/MatMul/ReadVariableOp?/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp?&vq_vae/vector_quantizer/ReadVariableOp?
+vq_vae/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp4vq_vae_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
vq_vae/encoder/conv2d/Conv2DConv2Dinput_43vq_vae/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
,vq_vae/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp5vq_vae_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
vq_vae/encoder/conv2d/BiasAddBiasAdd%vq_vae/encoder/conv2d/Conv2D:output:04vq_vae/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
vq_vae/encoder/conv2d/ReluRelu&vq_vae/encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
-vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
vq_vae/encoder/conv2d_1/Conv2DConv2D(vq_vae/encoder/conv2d/Relu:activations:05vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
.vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vq_vae/encoder/conv2d_1/BiasAddBiasAdd'vq_vae/encoder/conv2d_1/Conv2D:output:06vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
vq_vae/encoder/conv2d_1/ReluRelu(vq_vae/encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
-vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vq_vae/encoder/conv2d_2/Conv2DConv2D*vq_vae/encoder/conv2d_1/Relu:activations:05vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
.vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vq_vae/encoder/conv2d_2/BiasAddBiasAdd'vq_vae/encoder/conv2d_2/Conv2D:output:06vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
vq_vae/encoder/conv2d_2/ReluRelu(vq_vae/encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
-vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vq_vae/encoder/conv2d_3/Conv2DConv2D*vq_vae/encoder/conv2d_2/Relu:activations:05vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
.vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vq_vae/encoder/conv2d_3/BiasAddBiasAdd'vq_vae/encoder/conv2d_3/Conv2D:output:06vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
vq_vae/encoder/conv2d_3/ReluRelu(vq_vae/encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
-vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
vq_vae/encoder/conv2d_4/Conv2DConv2D*vq_vae/encoder/conv2d_3/Relu:activations:05vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
.vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vq_vae/encoder/conv2d_4/BiasAddBiasAdd'vq_vae/encoder/conv2d_4/Conv2D:output:06vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?u
vq_vae/vector_quantizer/ShapeShape(vq_vae/encoder/conv2d_4/BiasAdd:output:0*
T0*
_output_shapes
:v
%vq_vae/vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
vq_vae/vector_quantizer/ReshapeReshape(vq_vae/encoder/conv2d_4/BiasAdd:output:0.vq_vae/vector_quantizer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
-vq_vae/vector_quantizer/MatMul/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
vq_vae/vector_quantizer/MatMulMatMul(vq_vae/vector_quantizer/Reshape:output:05vq_vae/vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
vq_vae/vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vq_vae/vector_quantizer/powPow(vq_vae/vector_quantizer/Reshape:output:0&vq_vae/vector_quantizer/pow/y:output:0*
T0*(
_output_shapes
:??????????o
-vq_vae/vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
vq_vae/vector_quantizer/SumSumvq_vae/vector_quantizer/pow:z:06vq_vae/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(?
&vq_vae/vector_quantizer/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0d
vq_vae/vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vq_vae/vector_quantizer/pow_1Pow.vq_vae/vector_quantizer/ReadVariableOp:value:0(vq_vae/vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	?@q
/vq_vae/vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : ?
vq_vae/vector_quantizer/Sum_1Sum!vq_vae/vector_quantizer/pow_1:z:08vq_vae/vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:@?
vq_vae/vector_quantizer/addAddV2$vq_vae/vector_quantizer/Sum:output:0&vq_vae/vector_quantizer/Sum_1:output:0*
T0*'
_output_shapes
:?????????@b
vq_vae/vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vq_vae/vector_quantizer/mulMul&vq_vae/vector_quantizer/mul/x:output:0(vq_vae/vector_quantizer/MatMul:product:0*
T0*'
_output_shapes
:?????????@?
vq_vae/vector_quantizer/subSubvq_vae/vector_quantizer/add:z:0vq_vae/vector_quantizer/mul:z:0*
T0*'
_output_shapes
:?????????@j
(vq_vae/vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :?
vq_vae/vector_quantizer/ArgMinArgMinvq_vae/vector_quantizer/sub:z:01vq_vae/vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:?????????m
(vq_vae/vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??n
)vq_vae/vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    g
%vq_vae/vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :@?
vq_vae/vector_quantizer/one_hotOneHot'vq_vae/vector_quantizer/ArgMin:output:0.vq_vae/vector_quantizer/one_hot/depth:output:01vq_vae/vector_quantizer/one_hot/on_value:output:02vq_vae/vector_quantizer/one_hot/off_value:output:0*
T0*'
_output_shapes
:?????????@?
/vq_vae/vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
 vq_vae/vector_quantizer/MatMul_1MatMul(vq_vae/vector_quantizer/one_hot:output:07vq_vae/vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????*
transpose_b(?
!vq_vae/vector_quantizer/Reshape_1Reshape*vq_vae/vector_quantizer/MatMul_1:product:0&vq_vae/vector_quantizer/Shape:output:0*
T0*0
_output_shapes
:?????????@@??
$vq_vae/vector_quantizer/StopGradientStopGradient*vq_vae/vector_quantizer/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@@??
vq_vae/vector_quantizer/sub_1Sub-vq_vae/vector_quantizer/StopGradient:output:0(vq_vae/encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?d
vq_vae/vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vq_vae/vector_quantizer/pow_2Pow!vq_vae/vector_quantizer/sub_1:z:0(vq_vae/vector_quantizer/pow_2/y:output:0*
T0*0
_output_shapes
:?????????@@?v
vq_vae/vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
vq_vae/vector_quantizer/MeanMean!vq_vae/vector_quantizer/pow_2:z:0&vq_vae/vector_quantizer/Const:output:0*
T0*
_output_shapes
: ?
&vq_vae/vector_quantizer/StopGradient_1StopGradient(vq_vae/encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vq_vae/vector_quantizer/sub_2Sub*vq_vae/vector_quantizer/Reshape_1:output:0/vq_vae/vector_quantizer/StopGradient_1:output:0*
T0*0
_output_shapes
:?????????@@?d
vq_vae/vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
vq_vae/vector_quantizer/pow_3Pow!vq_vae/vector_quantizer/sub_2:z:0(vq_vae/vector_quantizer/pow_3/y:output:0*
T0*0
_output_shapes
:?????????@@?x
vq_vae/vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
vq_vae/vector_quantizer/Mean_1Mean!vq_vae/vector_quantizer/pow_3:z:0(vq_vae/vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: d
vq_vae/vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
vq_vae/vector_quantizer/mul_1Mul(vq_vae/vector_quantizer/mul_1/x:output:0%vq_vae/vector_quantizer/Mean:output:0*
T0*
_output_shapes
: ?
vq_vae/vector_quantizer/add_1AddV2!vq_vae/vector_quantizer/mul_1:z:0'vq_vae/vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: ?
vq_vae/vector_quantizer/sub_3Sub*vq_vae/vector_quantizer/Reshape_1:output:0(vq_vae/encoder/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
&vq_vae/vector_quantizer/StopGradient_2StopGradient!vq_vae/vector_quantizer/sub_3:z:0*
T0*0
_output_shapes
:?????????@@??
vq_vae/vector_quantizer/add_2AddV2(vq_vae/encoder/conv2d_4/BiasAdd:output:0/vq_vae/vector_quantizer/StopGradient_2:output:0*
T0*0
_output_shapes
:?????????@@?v
%vq_vae/decoder/conv2d_transpose/ShapeShape!vq_vae/vector_quantizer/add_2:z:0*
T0*
_output_shapes
:}
3vq_vae/decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5vq_vae/decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5vq_vae/decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-vq_vae/decoder/conv2d_transpose/strided_sliceStridedSlice.vq_vae/decoder/conv2d_transpose/Shape:output:0<vq_vae/decoder/conv2d_transpose/strided_slice/stack:output:0>vq_vae/decoder/conv2d_transpose/strided_slice/stack_1:output:0>vq_vae/decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'vq_vae/decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@i
'vq_vae/decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@i
'vq_vae/decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
%vq_vae/decoder/conv2d_transpose/stackPack6vq_vae/decoder/conv2d_transpose/strided_slice:output:00vq_vae/decoder/conv2d_transpose/stack/1:output:00vq_vae/decoder/conv2d_transpose/stack/2:output:00vq_vae/decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose/strided_slice_1StridedSlice.vq_vae/decoder/conv2d_transpose/stack:output:0>vq_vae/decoder/conv2d_transpose/strided_slice_1/stack:output:0@vq_vae/decoder/conv2d_transpose/strided_slice_1/stack_1:output:0@vq_vae/decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?vq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHvq_vae_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
0vq_vae/decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput.vq_vae/decoder/conv2d_transpose/stack:output:0Gvq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0!vq_vae/vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
6vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp?vq_vae_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
'vq_vae/decoder/conv2d_transpose/BiasAddBiasAdd9vq_vae/decoder/conv2d_transpose/conv2d_transpose:output:0>vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
$vq_vae/decoder/conv2d_transpose/ReluRelu0vq_vae/decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
'vq_vae/decoder/conv2d_transpose_1/ShapeShape2vq_vae/decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose_1/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_1/Shape:output:0>vq_vae/decoder/conv2d_transpose_1/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_1/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
'vq_vae/decoder/conv2d_transpose_1/stackPack8vq_vae/decoder/conv2d_transpose_1/strided_slice:output:02vq_vae/decoder/conv2d_transpose_1/stack/1:output:02vq_vae/decoder/conv2d_transpose_1/stack/2:output:02vq_vae/decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:?
7vq_vae/decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9vq_vae/decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9vq_vae/decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1vq_vae/decoder/conv2d_transpose_1/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_1/stack:output:0@vq_vae/decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Avq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
2vq_vae/decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_1/stack:output:0Ivq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:02vq_vae/decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
8vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
)vq_vae/decoder/conv2d_transpose_1/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_1/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
&vq_vae/decoder/conv2d_transpose_1/ReluRelu2vq_vae/decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
'vq_vae/decoder/conv2d_transpose_2/ShapeShape4vq_vae/decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose_2/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_2/Shape:output:0>vq_vae/decoder/conv2d_transpose_2/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_2/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
'vq_vae/decoder/conv2d_transpose_2/stackPack8vq_vae/decoder/conv2d_transpose_2/strided_slice:output:02vq_vae/decoder/conv2d_transpose_2/stack/1:output:02vq_vae/decoder/conv2d_transpose_2/stack/2:output:02vq_vae/decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:?
7vq_vae/decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9vq_vae/decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9vq_vae/decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1vq_vae/decoder/conv2d_transpose_2/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_2/stack:output:0@vq_vae/decoder/conv2d_transpose_2/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Avq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
2vq_vae/decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_2/stack:output:0Ivq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
8vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
)vq_vae/decoder/conv2d_transpose_2/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_2/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
&vq_vae/decoder/conv2d_transpose_2/ReluRelu2vq_vae/decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@?
'vq_vae/decoder/conv2d_transpose_3/ShapeShape4vq_vae/decoder/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose_3/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_3/Shape:output:0>vq_vae/decoder/conv2d_transpose_3/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
'vq_vae/decoder/conv2d_transpose_3/stackPack8vq_vae/decoder/conv2d_transpose_3/strided_slice:output:02vq_vae/decoder/conv2d_transpose_3/stack/1:output:02vq_vae/decoder/conv2d_transpose_3/stack/2:output:02vq_vae/decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:?
7vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1vq_vae/decoder/conv2d_transpose_3/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_3/stack:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
2vq_vae/decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_3/stack:output:0Ivq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
)vq_vae/decoder/conv2d_transpose_3/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_3/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
&vq_vae/decoder/conv2d_transpose_3/ReluRelu2vq_vae/decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
'vq_vae/decoder/conv2d_transpose_4/ShapeShape4vq_vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose_4/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_4/Shape:output:0>vq_vae/decoder/conv2d_transpose_4/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
'vq_vae/decoder/conv2d_transpose_4/stackPack8vq_vae/decoder/conv2d_transpose_4/strided_slice:output:02vq_vae/decoder/conv2d_transpose_4/stack/1:output:02vq_vae/decoder/conv2d_transpose_4/stack/2:output:02vq_vae/decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:?
7vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1vq_vae/decoder/conv2d_transpose_4/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_4/stack:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
2vq_vae/decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_4/stack:output:0Ivq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
)vq_vae/decoder/conv2d_transpose_4/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_4/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
&vq_vae/decoder/conv2d_transpose_4/ReluRelu2vq_vae/decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ?
'vq_vae/decoder/conv2d_transpose_5/ShapeShape4vq_vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/vq_vae/decoder/conv2d_transpose_5/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_5/Shape:output:0>vq_vae/decoder/conv2d_transpose_5/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)vq_vae/decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
'vq_vae/decoder/conv2d_transpose_5/stackPack8vq_vae/decoder/conv2d_transpose_5/strided_slice:output:02vq_vae/decoder/conv2d_transpose_5/stack/1:output:02vq_vae/decoder/conv2d_transpose_5/stack/2:output:02vq_vae/decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:?
7vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1vq_vae/decoder/conv2d_transpose_5/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_5/stack:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
2vq_vae/decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_5/stack:output:0Ivq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)vq_vae/decoder/conv2d_transpose_5/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_5/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
IdentityIdentity2vq_vae/decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp7^vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp@^vq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp-^vq_vae/encoder/conv2d/BiasAdd/ReadVariableOp,^vq_vae/encoder/conv2d/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOp.^vq_vae/vector_quantizer/MatMul/ReadVariableOp0^vq_vae/vector_quantizer/MatMul_1/ReadVariableOp'^vq_vae/vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????@@: : : : : : : : : : : : : : : : : : : : : : : 2p
6vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp6vq_vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2?
?vq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?vq_vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Avq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
Avq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2?
Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2?
Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2\
,vq_vae/encoder/conv2d/BiasAdd/ReadVariableOp,vq_vae/encoder/conv2d/BiasAdd/ReadVariableOp2Z
+vq_vae/encoder/conv2d/Conv2D/ReadVariableOp+vq_vae/encoder/conv2d/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_1/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_1/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_2/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_2/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_3/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_3/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_4/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_4/Conv2D/ReadVariableOp2^
-vq_vae/vector_quantizer/MatMul/ReadVariableOp-vq_vae/vector_quantizer/MatMul/ReadVariableOp2b
/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp2P
&vq_vae/vector_quantizer/ReadVariableOp&vq_vae/vector_quantizer/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_4
?

?
(__inference_encoder_layer_call_fn_125527
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_125479x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?[
?
"__inference__traced_restore_128180
file_prefix4
!assignvariableop_embeddings_vqvae:	?@:
 assignvariableop_1_conv2d_kernel: ,
assignvariableop_2_conv2d_bias: <
"assignvariableop_3_conv2d_1_kernel: @.
 assignvariableop_4_conv2d_1_bias:@<
"assignvariableop_5_conv2d_2_kernel:@@.
 assignvariableop_6_conv2d_2_bias:@<
"assignvariableop_7_conv2d_3_kernel:@@.
 assignvariableop_8_conv2d_3_bias:@=
"assignvariableop_9_conv2d_4_kernel:@?0
!assignvariableop_10_conv2d_4_bias:	?F
+assignvariableop_11_conv2d_transpose_kernel:@?7
)assignvariableop_12_conv2d_transpose_bias:@G
-assignvariableop_13_conv2d_transpose_1_kernel:@@9
+assignvariableop_14_conv2d_transpose_1_bias:@G
-assignvariableop_15_conv2d_transpose_2_kernel:@@9
+assignvariableop_16_conv2d_transpose_2_bias:@G
-assignvariableop_17_conv2d_transpose_3_kernel: @9
+assignvariableop_18_conv2d_transpose_3_bias: G
-assignvariableop_19_conv2d_transpose_4_kernel:  9
+assignvariableop_20_conv2d_transpose_4_bias: G
-assignvariableop_21_conv2d_transpose_5_kernel: 9
+assignvariableop_22_conv2d_transpose_5_bias:
identity_24??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_embeddings_vqvaeIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_conv2d_transpose_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_conv2d_transpose_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_conv2d_transpose_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_conv2d_transpose_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_conv2d_transpose_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_conv2d_transpose_3_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv2d_transpose_3_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_4_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_5_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_5_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?!
?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_127967

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_127752

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?&
?
C__inference_decoder_layer_call_and_return_conditional_losses_126074
input_32
conv2d_transpose_126043:@?%
conv2d_transpose_126045:@3
conv2d_transpose_1_126048:@@'
conv2d_transpose_1_126050:@3
conv2d_transpose_2_126053:@@'
conv2d_transpose_2_126055:@3
conv2d_transpose_3_126058: @'
conv2d_transpose_3_126060: 3
conv2d_transpose_4_126063:  '
conv2d_transpose_4_126065: 3
conv2d_transpose_5_126068: '
conv2d_transpose_5_126070:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_transpose_126043conv2d_transpose_126045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_125623?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_126048conv2d_transpose_1_126050*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_126053conv2d_transpose_2_126055*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_125713?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_126058conv2d_transpose_3_126060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_126063conv2d_transpose_4_126065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_126068conv2d_transpose_5_126070*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_125847?
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@?
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:?????????@@?: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????@@?
!
_user_specified_name	input_3
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_127713

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_125350

inputs'
conv2d_125277: 
conv2d_125279: )
conv2d_1_125294: @
conv2d_1_125296:@)
conv2d_2_125311:@@
conv2d_2_125313:@)
conv2d_3_125328:@@
conv2d_3_125330:@*
conv2d_4_125344:@?
conv2d_4_125346:	?
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_125277conv2d_125279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_125276?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_125294conv2d_1_125296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_125293?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_125311conv2d_2_125313*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_125310?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_125328conv2d_3_125330*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_125327?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_125344conv2d_4_125346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????@@: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_4_layer_call_fn_127742

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_125343x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_125803

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_3_layer_call_fn_127890

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_125758?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_1_layer_call_fn_127804

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_125668?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_48
serving_default_input_4:0?????????@@C
decoder8
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
?
 layer-0
!layer_with_weights-0
!layer-1
"layer_with_weights-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer_with_weights-4
%layer-5
&layer_with_weights-5
&layer-6
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_network
?
-0
.1
/2
03
14
25
36
47
58
69
10
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22"
trackable_list_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
10
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32?
'__inference_vq_vae_layer_call_fn_126269
'__inference_vq_vae_layer_call_fn_126704
'__inference_vq_vae_layer_call_fn_126756
'__inference_vq_vae_layer_call_fn_126489?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
?
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126960
B__inference_vq_vae_layer_call_and_return_conditional_losses_127164
B__inference_vq_vae_layer_call_and_return_conditional_losses_126544
B__inference_vq_vae_layer_call_and_return_conditional_losses_126599?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
?B?
!__inference__wrapped_model_125258input_4"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Pserving_default"
signature_map
"
_tf_keras_input_layer
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

-kernel
.bias
 W_jit_compiled_convolution_op"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

/kernel
0bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

1kernel
2bias
 e_jit_compiled_convolution_op"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

3kernel
4bias
 l_jit_compiled_convolution_op"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

5kernel
6bias
 s_jit_compiled_convolution_op"
_tf_keras_layer
f
-0
.1
/2
03
14
25
36
47
58
69"
trackable_list_wrapper
f
-0
.1
/2
03
14
25
36
47
58
69"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
ytrace_0
ztrace_1
{trace_2
|trace_32?
(__inference_encoder_layer_call_fn_125373
(__inference_encoder_layer_call_fn_127189
(__inference_encoder_layer_call_fn_127214
(__inference_encoder_layer_call_fn_125527?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zytrace_0zztrace_1z{trace_2z|trace_3
?
}trace_0
~trace_1
trace_2
?trace_32?
C__inference_encoder_layer_call_and_return_conditional_losses_127252
C__inference_encoder_layer_call_and_return_conditional_losses_127290
C__inference_encoder_layer_call_and_return_conditional_losses_125556
C__inference_encoder_layer_call_and_return_conditional_losses_125585?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z}trace_0z~trace_1ztrace_2z?trace_3
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_vector_quantizer_layer_call_fn_127298?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_127349?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!	?@2embeddings_vqvae
"
_tf_keras_input_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

7kernel
8bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

9kernel
:bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

;kernel
<bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

=kernel
>bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

?kernel
@bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

Akernel
Bbias
!?_jit_compiled_convolution_op"
_tf_keras_layer
v
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11"
trackable_list_wrapper
v
70
81
92
:3
;4
<5
=6
>7
?8
@9
A10
B11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
(__inference_decoder_layer_call_fn_125919
(__inference_decoder_layer_call_fn_127378
(__inference_decoder_layer_call_fn_127407
(__inference_decoder_layer_call_fn_126040?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
C__inference_decoder_layer_call_and_return_conditional_losses_127530
C__inference_decoder_layer_call_and_return_conditional_losses_127653
C__inference_decoder_layer_call_and_return_conditional_losses_126074
C__inference_decoder_layer_call_and_return_conditional_losses_126108?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
':% 2conv2d/kernel
: 2conv2d/bias
):' @2conv2d_1/kernel
:@2conv2d_1/bias
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
*:(@?2conv2d_4/kernel
:?2conv2d_4/bias
2:0@?2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
3:1@@2conv2d_transpose_1/kernel
%:#@2conv2d_transpose_1/bias
3:1@@2conv2d_transpose_2/kernel
%:#@2conv2d_transpose_2/bias
3:1 @2conv2d_transpose_3/kernel
%:# 2conv2d_transpose_3/bias
3:1  2conv2d_transpose_4/kernel
%:# 2conv2d_transpose_4/bias
3:1 2conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_vq_vae_layer_call_fn_126269input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_vq_vae_layer_call_fn_126704inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_vq_vae_layer_call_fn_126756inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_vq_vae_layer_call_fn_126489input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126960inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_vq_vae_layer_call_and_return_conditional_losses_127164inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126544input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126599input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_126652input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_conv2d_layer_call_fn_127662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_conv2d_layer_call_and_return_conditional_losses_127673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_1_layer_call_fn_127682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_127693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_2_layer_call_fn_127702?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_127713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_3_layer_call_fn_127722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_127733?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_4_layer_call_fn_127742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_127752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_encoder_layer_call_fn_125373input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_encoder_layer_call_fn_127189inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_encoder_layer_call_fn_127214inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_encoder_layer_call_fn_125527input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoder_layer_call_and_return_conditional_losses_127252inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoder_layer_call_and_return_conditional_losses_127290inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoder_layer_call_and_return_conditional_losses_125556input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_encoder_layer_call_and_return_conditional_losses_125585input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_vector_quantizer_layer_call_fn_127298x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_127349x"?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_conv2d_transpose_layer_call_fn_127761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_127795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_1_layer_call_fn_127804?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_127838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_2_layer_call_fn_127847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_127881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_3_layer_call_fn_127890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_127924?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_4_layer_call_fn_127933?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_127967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_5_layer_call_fn_127976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_128009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
Q
 0
!1
"2
#3
$4
%5
&6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_decoder_layer_call_fn_125919input_3"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_127378inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_127407inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_126040input_3"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_127530inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_127653inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_126074input_3"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_126108input_3"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_conv2d_layer_call_fn_127662inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_conv2d_layer_call_and_return_conditional_losses_127673inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_1_layer_call_fn_127682inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_127693inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_2_layer_call_fn_127702inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_127713inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_3_layer_call_fn_127722inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_127733inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_4_layer_call_fn_127742inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_127752inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_conv2d_transpose_layer_call_fn_127761inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_127795inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_conv2d_transpose_1_layer_call_fn_127804inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_127838inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_conv2d_transpose_2_layer_call_fn_127847inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_127881inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_conv2d_transpose_3_layer_call_fn_127890inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_127924inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_conv2d_transpose_4_layer_call_fn_127933inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_127967inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_conv2d_transpose_5_layer_call_fn_127976inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_128009inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_125258?-./0123456789:;<=>?@AB8?5
.?+
)?&
input_4?????????@@
? "9?6
4
decoder)?&
decoder?????????@@?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_127693l/07?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_1_layer_call_fn_127682_/07?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@@?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_127713l127?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_2_layer_call_fn_127702_127?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_127733l347?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_3_layer_call_fn_127722_347?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_127752m567?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????@@?
? ?
)__inference_conv2d_4_layer_call_fn_127742`567?4
-?*
(?%
inputs?????????@@@
? "!??????????@@??
B__inference_conv2d_layer_call_and_return_conditional_losses_127673l-.7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@ 
? ?
'__inference_conv2d_layer_call_fn_127662_-.7?4
-?*
(?%
inputs?????????@@
? " ??????????@@ ?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_127838?9:I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_conv2d_transpose_1_layer_call_fn_127804?9:I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_127881?;<I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_conv2d_transpose_2_layer_call_fn_127847?;<I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_127924?=>I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_3_layer_call_fn_127890?=>I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_127967??@I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_4_layer_call_fn_127933??@I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_128009?ABI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_5_layer_call_fn_127976?ABI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_127795?78J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
1__inference_conv2d_transpose_layer_call_fn_127761?78J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
C__inference_decoder_layer_call_and_return_conditional_losses_126074?789:;<=>?@ABA?>
7?4
*?'
input_3?????????@@?
p 

 
? "-?*
#? 
0?????????@@
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_126108?789:;<=>?@ABA?>
7?4
*?'
input_3?????????@@?
p

 
? "-?*
#? 
0?????????@@
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_127530789:;<=>?@AB@?=
6?3
)?&
inputs?????????@@?
p 

 
? "-?*
#? 
0?????????@@
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_127653789:;<=>?@AB@?=
6?3
)?&
inputs?????????@@?
p

 
? "-?*
#? 
0?????????@@
? ?
(__inference_decoder_layer_call_fn_125919s789:;<=>?@ABA?>
7?4
*?'
input_3?????????@@?
p 

 
? " ??????????@@?
(__inference_decoder_layer_call_fn_126040s789:;<=>?@ABA?>
7?4
*?'
input_3?????????@@?
p

 
? " ??????????@@?
(__inference_decoder_layer_call_fn_127378r789:;<=>?@AB@?=
6?3
)?&
inputs?????????@@?
p 

 
? " ??????????@@?
(__inference_decoder_layer_call_fn_127407r789:;<=>?@AB@?=
6?3
)?&
inputs?????????@@?
p

 
? " ??????????@@?
C__inference_encoder_layer_call_and_return_conditional_losses_125556~
-./0123456@?=
6?3
)?&
input_1?????????@@
p 

 
? ".?+
$?!
0?????????@@?
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_125585~
-./0123456@?=
6?3
)?&
input_1?????????@@
p

 
? ".?+
$?!
0?????????@@?
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_127252}
-./0123456??<
5?2
(?%
inputs?????????@@
p 

 
? ".?+
$?!
0?????????@@?
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_127290}
-./0123456??<
5?2
(?%
inputs?????????@@
p

 
? ".?+
$?!
0?????????@@?
? ?
(__inference_encoder_layer_call_fn_125373q
-./0123456@?=
6?3
)?&
input_1?????????@@
p 

 
? "!??????????@@??
(__inference_encoder_layer_call_fn_125527q
-./0123456@?=
6?3
)?&
input_1?????????@@
p

 
? "!??????????@@??
(__inference_encoder_layer_call_fn_127189p
-./0123456??<
5?2
(?%
inputs?????????@@
p 

 
? "!??????????@@??
(__inference_encoder_layer_call_fn_127214p
-./0123456??<
5?2
(?%
inputs?????????@@
p

 
? "!??????????@@??
$__inference_signature_wrapper_126652?-./0123456789:;<=>?@ABC?@
? 
9?6
4
input_4)?&
input_4?????????@@"9?6
4
decoder)?&
decoder?????????@@?
L__inference_vector_quantizer_layer_call_and_return_conditional_losses_127349v3?0
)?&
$?!
x?????????@@?
? "<?9
$?!
0?????????@@?
?
?	
1/0 ?
1__inference_vector_quantizer_layer_call_fn_127298[3?0
)?&
$?!
x?????????@@?
? "!??????????@@??
B__inference_vq_vae_layer_call_and_return_conditional_losses_126544?-./0123456789:;<=>?@AB@?=
6?3
)?&
input_4?????????@@
p 

 
? ";?8
#? 
0?????????@@
?
?	
1/0 ?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126599?-./0123456789:;<=>?@AB@?=
6?3
)?&
input_4?????????@@
p

 
? ";?8
#? 
0?????????@@
?
?	
1/0 ?
B__inference_vq_vae_layer_call_and_return_conditional_losses_126960?-./0123456789:;<=>?@AB??<
5?2
(?%
inputs?????????@@
p 

 
? ";?8
#? 
0?????????@@
?
?	
1/0 ?
B__inference_vq_vae_layer_call_and_return_conditional_losses_127164?-./0123456789:;<=>?@AB??<
5?2
(?%
inputs?????????@@
p

 
? ";?8
#? 
0?????????@@
?
?	
1/0 ?
'__inference_vq_vae_layer_call_fn_126269}-./0123456789:;<=>?@AB@?=
6?3
)?&
input_4?????????@@
p 

 
? " ??????????@@?
'__inference_vq_vae_layer_call_fn_126489}-./0123456789:;<=>?@AB@?=
6?3
)?&
input_4?????????@@
p

 
? " ??????????@@?
'__inference_vq_vae_layer_call_fn_126704|-./0123456789:;<=>?@AB??<
5?2
(?%
inputs?????????@@
p 

 
? " ??????????@@?
'__inference_vq_vae_layer_call_fn_126756|-./0123456789:;<=>?@AB??<
5?2
(?%
inputs?????????@@
p

 
? " ??????????@@