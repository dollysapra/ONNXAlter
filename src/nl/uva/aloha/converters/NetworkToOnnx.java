package nl.uva.aloha.converters;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.google.protobuf.ByteString;

import espam.datamodel.graph.cnn.BoundaryMode;
import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.cnn.Neuron;
import espam.datamodel.graph.cnn.connections.Connection;
import espam.datamodel.graph.cnn.neurons.arithmetic.Arithmetic;
import espam.datamodel.graph.cnn.neurons.cnn.CNNNeuron;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.cnn.neurons.cnn.Pooling;
import espam.datamodel.graph.cnn.neurons.neurontypes.NonLinearType;
import espam.datamodel.graph.cnn.neurons.neurontypes.PoolingType;
import espam.datamodel.graph.cnn.neurons.simple.DenseBlock;
import espam.datamodel.graph.cnn.neurons.simple.NonLinear;
import espam.datamodel.graph.csdf.datasctructures.Tensor;
//import espam.datamodel.graph.sdf.datasctructures.Tensor;
import onnx.ONNX;
//import espam.datamodel.onnx.ONNX;
import  onnx.ONNX.AttributeProto;
import  onnx.ONNX.AttributeProto.AttributeType;
import  onnx.ONNX.GraphProto;
import  onnx.ONNX.ModelProto;
import  onnx.ONNX.NodeProto;
import  onnx.ONNX.OperatorSetIdProto;
import  onnx.ONNX.TensorProto;
import  onnx.ONNX.TensorProto.DataType;
import  onnx.ONNX.TensorShapeProto;
import  onnx.ONNX.TensorShapeProto.Dimension;
import  onnx.ONNX.TypeProto;
import  onnx.ONNX.ValueInfoProto;


/**
 * 
 * @author Dolly Sapra
 *
 * This class converts espam "Network" class to an ONNX.
 * 
 */
public class NetworkToOnnx {

	
	private Network _network;
	
	private GraphProto _graphProto;
	
	public static String DATASET = "CIFAR-10";
	public static Boolean BatchNormAfterConv = true;
	
	public NetworkToOnnx(Network network)
	{
		_network = network;
		
	}
	
	public Network getNetwork()
	{
		return _network;
	}
	
	public ModelProto convertToONNXModel()
	{
		if(_network == null)
		{
			System.err.println("Network is not set properly. Nothing to convert");
			return null;
		}
		
		if(!_network.checkConsistency())
		{
			System.err.println("Network is not consistent. Cannot convert");
			return null;
		}
		
		 Network network = _network;
		
		 ModelProto.Builder _modelBuilder = ModelProto.newBuilder();
		 GraphProto.Builder _graphBuilder = GraphProto.newBuilder();
		
		
		//NODE creation
		Iterator<Layer> itr = network.getLayers().iterator();
		while(itr.hasNext()) 
		{
			Layer l = itr.next();
			Neuron currentNeuron = l.getNeuron();
			ArrayList<String> layerInputs = new ArrayList<String>(); 
			ArrayList<String> layerOutputs = new ArrayList<String>(); 
			//System.out.print(l.getName() + ":" + l.getInputFormat().toString());
			//Assuming input connections and output connections are same and Layer names are unique!!
			for(int i=0;i<l.getInputConnections().size();i++)
			{
				Connection cnxn = l.getInputConnections().get(i);
				if(cnxn.getSrc().equals(network.getInputLayer()))
					layerInputs.add("input_data");
				else
					layerInputs.add(cnxn.getSrcName() + "_output" );
			}
			
			for(int i=0;i<l.getOutputConnections().size();i++)
			{
				Connection cnxn = l.getOutputConnections().get(i);
				if(!layerOutputs.contains(cnxn.getSrcName() + "_output"))
					layerOutputs.add(cnxn.getSrcName() + "_output" );
			}
			
			if(l.equals(network.getInputLayer()))
			{
				Tensor opFrmt = new Tensor(l.getOutputFormat());
				opFrmt.addDimension(l.getNeuronsNum());
				//l.getName() + "_"+l.getOutputConnections().get(0).getDestName()
				ValueInfoProto valueProto = null;
				if(DATASET.equals("PAMAP2"))
					valueProto = createInputProto("input_data", new Tensor(1,40,100,1 ));
				else
					valueProto = createInputProto("input_data", Tensor.reverse(opFrmt));
				_graphBuilder.addInput(valueProto);
				
				continue;
			}
			else if(l.equals(network.getOutputLayer()))
			{
				
				Tensor outputFormat = new Tensor(10);
				if(DATASET.equals("PAMAP2"))
					outputFormat = new Tensor(12);
				outputFormat.addDimension(l.getNeuronsNum());
				
				ValueInfoProto valueProto = createInputProto(l.getInputConnections().get(0).getSrc().getName() + "_" + l.getName(), Tensor.reverse(outputFormat));
				_graphBuilder.addOutput(valueProto);
				
				continue;
			}
			else if(currentNeuron instanceof Convolution)
			{
				String weightName = l.getName()+"weights";
				String biasName = l.getName()+"bias";
				layerInputs.add(weightName);
				layerInputs.add(biasName);
				
				//TODO: Assuming only one inputconnection as of now - later extend it to sum of all inputs
				int kw = ((Convolution)(currentNeuron)).getKernelW();
				int kh = ((Convolution)(currentNeuron)).getKernelH();
				int inputConnectionsCount = l.getInputFormat().getLastDimSize();
				Tensor weightFormat = new Tensor(l.getNeuronsNum(),inputConnectionsCount,kw,kh);
				//if(GAMain.DATASET.equals("PAMAP2"))
					//weightFormat = new Tensor(l.getNeuronsNum(),inputConnectionsCount,kw);
				_graphBuilder.addInput(createInputProto( weightName,weightFormat));
				_graphBuilder.addInput(createInputProto(biasName, new Tensor(l.getNeuronsNum())) );
				
				_graphBuilder.addInitializer(createOrthoWeights(weightName, weightFormat)); //TODO:CONFIG
				_graphBuilder.addInitializer(createHeWeights(biasName, new Tensor(l.getNeuronsNum())) );
				
				if(BatchNormAfterConv)
				{
					ArrayList<String> intermediateInputOutput = new ArrayList<String>(); 
					String intermediator = l.getName()+"interBN";
					String BNname = l.getName() + "BN";
					intermediateInputOutput.add(intermediator);
					
					_graphBuilder.addNode(layerToOnnxNode(l,layerInputs,intermediateInputOutput));
					
					NodeProto.Builder BNBuilder = NodeProto.newBuilder();
					BNBuilder.setOpType("BatchNormalization");
					BNBuilder.setName(BNname );
					BNBuilder.addAttribute(floatValueToAttribute("epsilon", (float)0.00001));
					BNBuilder.addAttribute(floatValueToAttribute("momentum", (float)0.9));
					
					BNBuilder.addInput(intermediator);
					BNBuilder.addInput(BNname + "_gamma");
					BNBuilder.addInput(BNname + "_beta");
					BNBuilder.addInput(BNname + "_mean");
					BNBuilder.addInput(BNname + "_var");
					
					Tensor format = new Tensor(l.getNeuronsNum());
					
					_graphBuilder.addInput(createInputProto(BNname + "_gamma", format));
					_graphBuilder.addInput(createInputProto(BNname + "_beta", format));
					_graphBuilder.addInput(createInputProto(BNname + "_mean", format));
					_graphBuilder.addInput(createInputProto(BNname + "_var", format));
				   
					
					_graphBuilder.addInitializer(createRandomWeight(BNname + "_gamma", format));
					_graphBuilder.addInitializer(createRandomWeight(BNname + "_beta", format));
					_graphBuilder.addInitializer(createZeroWeights(BNname + "_mean", format));
					_graphBuilder.addInitializer(createOneValuedWeights(BNname + "_var", format));
					
					BNBuilder.addOutput(layerOutputs.get(0));
					_graphBuilder.addNode(BNBuilder.build());
				}
				else
					_graphBuilder.addNode(layerToOnnxNode(l,layerInputs,layerOutputs));
			}
			else if(currentNeuron instanceof DenseBlock)
			{
				String weightName = l.getName()+"weights";
				String biasName = l.getName()+"bias";
				layerInputs.add(weightName);
				layerInputs.add(biasName);
				
				//Our Format is  W x H x N1 x N2,    N2 - Number of neurons in previous layer. N1 is ignored in gemm layer, But i do not know why yet ~Dolly 
				//OR it vanishes in one of the previous layers.. Conv/Relu/Pool - one of them absorbs it i think :)
				//ONNX Tensor order: N x C x H x W,     N- batch size, C- Channels
				Tensor ip = l.getInputFormat();
				int inputSize = ip.getDimSize(0);
				switch(ip.getDimensionality())
				{
		
				case 2: inputSize*=ip.getDimSize(1);
						break;
				case 3: 		
				case 4: inputSize*=(ip.getDimSize(1)*ip.getLastDimSize());
						break;
				}
				
				Tensor weightFormat = new Tensor(l.getOutputFormat().getLastDimSize(),inputSize );
				
				/*if((currentNeuron instanceof Gemm)&&((Gemm)(currentNeuron)).isBTransposed())
					weightFormat = Tensor.reverse(weightFormat);*/
				
				_graphBuilder.addInput(createInputProto( weightName, weightFormat));
				_graphBuilder.addInput(createInputProto(biasName, new Tensor(l.getOutputFormat().getLastDimSize())) );
				
				_graphBuilder.addInitializer(createHeWeights(weightName, weightFormat));
				_graphBuilder.addInitializer(createHeWeights(biasName, new Tensor(l.getOutputFormat().getLastDimSize())) );
				
				_graphBuilder.addNode(layerToOnnxNode(l,layerInputs,layerOutputs));
			}
			
			//Other than inp/outp/conv/dense layers - such as Relu/Activations/softmax.
			else 
				_graphBuilder.addNode(layerToOnnxNode(l,layerInputs,layerOutputs));
				
		}
	
		_modelBuilder.setProducerName("ALOHA");
		_modelBuilder.addOpsetImport(setopImportSet());
		_modelBuilder.setIrVersion(ONNX.Version.IR_VERSION.getNumber());
		
		_graphBuilder.setName("GAtest"+new Date().getTime());
		_graphProto = _graphBuilder.build();
		_modelBuilder.setGraph(_graphProto);
		
		
		return(_modelBuilder.build());
		
	}
	
	private OperatorSetIdProto setopImportSet()
	{
		OperatorSetIdProto.Builder _opSetBuilder = OperatorSetIdProto.newBuilder();
		_opSetBuilder.setDomain("");
		_opSetBuilder.setVersion(9);
		return _opSetBuilder.build();
	}
	
	public static ValueInfoProto createInputProto(String name, Tensor inputTensorDims)
	{
		ValueInfoProto.Builder inputBuilder = ValueInfoProto.newBuilder();
		inputBuilder.setName(name);
		
		
		TypeProto.Builder typeBuilder = TypeProto.newBuilder();
		
		TensorShapeProto.Builder shapeBuilder = TensorShapeProto.newBuilder();
		
		//inputTensorDims = Tensor.reverse(inputTensorDims);
		for(int i=0; i< inputTensorDims.getDimensionality(); i++)
		{
			Dimension.Builder d =  Dimension.newBuilder();
			d.setDimValue(inputTensorDims.getDimSize(i));
			shapeBuilder.addDim(d.build());
		}
		TypeProto.Tensor.Builder tensorBuilder =  TypeProto.Tensor.newBuilder();
		tensorBuilder.setElemType(DataType.FLOAT);
		tensorBuilder.setShape(shapeBuilder.build());
	
		
		typeBuilder.setTensorType(tensorBuilder.build());
		inputBuilder.setType(typeBuilder.build());
		return inputBuilder.build();
		
	}
	
	
	public static TensorProto createZeroWeights(String name, Tensor inputTensorDims)
	{
		TensorProto.Builder _builder = TensorProto.newBuilder();
		
		_builder.setName(name);
		int totalValues =1;
		
		for(int i=0; i< inputTensorDims.getDimensionality(); i++)
		{
			int dim = inputTensorDims.getDimSize(i);
			_builder.addDims(dim);
			totalValues*=dim;
		}
		_builder.setDataType(DataType.FLOAT);
		
		ByteBuffer bb = ByteBuffer.allocate(totalValues*4);
		_builder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
		return _builder.build();
	}
	
	public static TensorProto createOneValuedWeights(String name, Tensor inputTensorDims)
	{
		TensorProto.Builder _builder = TensorProto.newBuilder();
		
		_builder.setName(name);
		int totalValues =1;
		
		for(int i=0; i< inputTensorDims.getDimensionality(); i++)
		{
			int dim = inputTensorDims.getDimSize(i);
			_builder.addDims(dim);
			totalValues*=dim;
		}
		_builder.setDataType(DataType.FLOAT);
		
		ByteBuffer bb = ByteBuffer.allocate(totalValues*4);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		for(int j=0; j< totalValues; j++)
			bb.putFloat(j*4, (float)1.0);
		_builder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
		return _builder.build();
	}
	
	public static TensorProto createRandomWeight(String name, Tensor inputTensorDims)
	{
		TensorProto.Builder _builder = TensorProto.newBuilder();
		
		_builder.setName(name);
		int totalValues =1;
		
		for(int i=0; i< inputTensorDims.getDimensionality(); i++)
		{
			int dim = inputTensorDims.getDimSize(i);
			_builder.addDims(dim);
			totalValues*=dim;
		}
		_builder.setDataType(DataType.FLOAT);
		
		ByteBuffer bb = ByteBuffer.allocate(totalValues*4);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		for(int j=0; j< totalValues; j++)
		{
			bb.putFloat(j*4, new Float(Math.random()*0.01));
			//bb.putFloat (j*4, (float)(new Random().nextGaussian()) * (float)Math.sqrt(2.0/totalValues));
		}
		_builder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
		return _builder.build();
	}
	
	
	public static TensorProto createHeWeights(String name, Tensor inputTensorDims)
	{
		TensorProto.Builder _builder = TensorProto.newBuilder();
		
		_builder.setName(name);
		int totalValues =1;
		
		for(int i=0; i< inputTensorDims.getDimensionality(); i++)
		{
			int dim = inputTensorDims.getDimSize(i);
			_builder.addDims(dim);
			totalValues*=dim;
		}
		_builder.setDataType(DataType.FLOAT);
		
		ByteBuffer bb = ByteBuffer.allocate(totalValues*4);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		for(int j=0;j<totalValues;j++)
		{
			float a;
			
			
			if(inputTensorDims.getDimensionality()>1)
			{
				 a = new Float(Math.random()*0.01);
				 if(Math.random() < 0.25)
						a*=-1;
				 if(inputTensorDims.getDimensionality() == 2)
					 a = (float)(new Random().nextGaussian()) * (float)Math.sqrt(2.0/inputTensorDims.getDimSize(1));
				 else if(inputTensorDims.getDimensionality() == 4)
					 a = (float)(new Random().nextGaussian()) * (float)Math.sqrt(2.0*inputTensorDims.getDimSize(0)/totalValues);
				 
			}
				
			else
			{
				a = (float)0.01;
				//new Float(Math.random()*0.01);
				//if(Math.random() < 0.25)
				//	a*=-1;
			}
			
			bb.putFloat(j*4, a);
		}
		_builder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
		return _builder.build();
	}
	
	public static TensorProto createOrthoWeights(String name, Tensor inputTensorDims)
	{
		TensorProto.Builder _builder = TensorProto.newBuilder();
		
		_builder.setName(name);
		int secondDims =1;
		int firstdim = inputTensorDims.getDimSize(0);
		_builder.addDims(firstdim);
		for(int i=1; i< inputTensorDims.getDimensionality(); i++)
		{
			int dim = inputTensorDims.getDimSize(i);
			_builder.addDims(dim);
			secondDims*=dim;
		}
		
		int totalValues = firstdim*secondDims;
		
		_builder.setDataType(DataType.FLOAT);
	    
		
		double[][] matrix = new double[firstdim][secondDims];
		for(int j=0; j<firstdim; j++)
		{
			for(int k=0; k<secondDims;k++)
				matrix[j][k] = new Float(Math.random());
		}
		Array2DRowRealMatrix rmt = new Array2DRowRealMatrix(matrix);
		SingularValueDecomposition SVD = new SingularValueDecomposition(rmt);
		RealMatrix vt =  SVD.getVT();
		RealMatrix U =  SVD.getU();
		double[][] matrixresult;
		if((U.getRowDimension() == firstdim) &&(U.getColumnDimension() == secondDims))
			matrixresult =  U.getData();
		else
			matrixresult = vt.getData();
		
		ByteBuffer bb = ByteBuffer.allocate(totalValues*4);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		int i =0;
		for(int j=0; j<firstdim; j++)
	      for(int k=0; k<secondDims;k++)	
			bb.putFloat(i++*4, (float)(matrixresult[j][k] * Math.sqrt(2.0)));
		
		
		_builder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
		return _builder.build();
	}
	
	
	//TODO Add inputs and outputs list
	public static NodeProto layerToOnnxNode(Layer l, ArrayList<String> inputs, ArrayList<String> outputs)
	{
		
		NodeProto.Builder _nodeBuilder = NodeProto.newBuilder();
		Neuron currentNeuron = l.getNeuron();
		_nodeBuilder.setName(l.getName());
		
		for(int inp =0; inp<inputs.size(); inp++)
		{ 
			_nodeBuilder.addInput( inputs.get(inp));
		}
		
		for(int outp=0; outp<outputs.size(); outp++)
		{ 
			_nodeBuilder.addOutput( outputs.get(outp));
		}
		
		if(currentNeuron instanceof CNNNeuron)
		{
			
			if(currentNeuron instanceof Pooling)
			{
				//Neuron name is type of pooling
				//layername is nodename
				
				if(currentNeuron.getName().contains("GLOBAL"))
				{
					if(currentNeuron.getName().contains(PoolingType.GLOBALLPPOOL.toString()))
						_nodeBuilder.setOpType("GlobalLpPool");
					else if(currentNeuron.getName().contains(PoolingType.GLOBALMAXPOOL.toString()))
						_nodeBuilder.setOpType("GlobalMaxPool");
					else
						_nodeBuilder.setOpType("GlobalAveragePool");
				}
					
				else 
				{
					if(currentNeuron.getName().contains(PoolingType.AVGPOOL.toString()))
						_nodeBuilder.setOpType("AveragePool");
					else //Default is MaxPool
						_nodeBuilder.setOpType("MaxPool");
					
					if(DATASET.equals("PAMAP2"))
					{
						_nodeBuilder.addAttribute(kernelShapeToAttributes1D (((CNNNeuron)(currentNeuron)).getKernelSize()) );
						_nodeBuilder.addAttribute(strideToAttributes1D(  ((CNNNeuron)(currentNeuron)).getStride() ));
						_nodeBuilder.addAttribute(padsToAttributes(l.getPads()));
					}
					else
					{
						_nodeBuilder.addAttribute(kernelShapeToAttributes (((CNNNeuron)(currentNeuron)).getKernelSize()) );
						_nodeBuilder.addAttribute(strideToAttributes(  ((CNNNeuron)(currentNeuron)).getStride()  ));
						_nodeBuilder.addAttribute(padsToAttributes(l.getPads()));
					}
					
					
				}
			}
			else if(currentNeuron instanceof Convolution)
			{
				_nodeBuilder.setOpType("Conv");
				if(DATASET.equals("PAMAP2"))
				{
					_nodeBuilder.addAttribute(kernelShapeToAttributes1D (((CNNNeuron)(currentNeuron)).getKernelSize()) );
					_nodeBuilder.addAttribute(strideToAttributes1D(  ((CNNNeuron)(currentNeuron)).getStride()  ));
					_nodeBuilder.addAttribute(padsToAttributes(l.getPads()));
				}
				else
				{
					_nodeBuilder.addAttribute(kernelShapeToAttributes (((CNNNeuron)(currentNeuron)).getKernelSize()) );
					_nodeBuilder.addAttribute(strideToAttributes(  ((CNNNeuron)(currentNeuron)).getStride()  ));
					_nodeBuilder.addAttribute(padsToAttributes(l.getPads()));
				}
				
				_nodeBuilder.addAttribute(boundaryToAttributes( ((CNNNeuron)(currentNeuron)).getBoundaryMode() ));
				
			}
			
			

			
		}
		
		
		//GEMM has alpha/beta values. Y== alpha*(A*B) + Beta*C. A,B,C are matrices. alpha,beta are just one number(float)
		//MatMul is only matrix multiplication
		/*else if(currentNeuron instanceof Gemm)
		{
			Gemm gemm = (Gemm)currentNeuron;
			_nodeBuilder.setOpType("Gemm");
			//broadcast has been removed from latest operators list
			//_nodeBuilder.addAttribute(intValueToAttribute("broadcast", (gemm.isBroadcast()?1:0) ));
			_nodeBuilder.addAttribute(intValueToAttribute("transB", (gemm.isBTransposed()?1:0)  ));
			_nodeBuilder.addAttribute(intValueToAttribute("transA", (gemm.isATransposed()?1:0)  ));
			_nodeBuilder.addAttribute(floatValueToAttribute("alpha", gemm.getAlpha()));
			_nodeBuilder.addAttribute(floatValueToAttribute("beta", gemm.getBeta()));
		}*/
		
		else if(currentNeuron instanceof DenseBlock)
		{
			int mn=1;
			_nodeBuilder.setOpType("Gemm");
			_nodeBuilder.addAttribute(floatValueToAttribute("alpha", new Float(1.0)));
			_nodeBuilder.addAttribute(floatValueToAttribute("beta", new Float(1.0)));
			_nodeBuilder.addAttribute(intValueToAttribute("transB", mn));
		}
		else if(currentNeuron instanceof NonLinear)
		{
			if(currentNeuron.getName().contains(NonLinearType.SOFTMAX.toString()))
				_nodeBuilder.setOpType("Softmax");
			else if(currentNeuron.getName().contains(NonLinearType.SOFTPLUS.toString()))
				_nodeBuilder.setOpType("Softplus");
			else if(currentNeuron.getName().contains(NonLinearType.THN.toString()))
				_nodeBuilder.setOpType("Tanh");
			else if(currentNeuron.getName().contains(NonLinearType.SIGM.toString()))
				_nodeBuilder.setOpType("Sigmoid");
			else if(currentNeuron.getName().contains(NonLinearType.SELU.toString()))
				_nodeBuilder.setOpType("Selu");
			else if(currentNeuron.getName().contains(NonLinearType.LeakyReLu.toString()))
				_nodeBuilder.setOpType("LeakyRelu");
			
			//Default non-linear neuron to Relu for now
			else //(currentNeuron.getName().contains(NonLinearType.ReLU.toString()))
				_nodeBuilder.setOpType("Relu");
			 
		}
		else if(currentNeuron instanceof Arithmetic)
		{
			_nodeBuilder.setOpType("Add");
		}
		
		
		return _nodeBuilder.build();
	}
	
	
	public static AttributeProto padsToAttributes1D(int[] pads)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("pads");
		_attBuilder.setType(AttributeType.INTS);
		
		if(pads== null)
		{
			pads = new int[] {0,0};
		}
		
		for(int i=0;i<pads.length;i+=2)
           _attBuilder.addInts(pads[i]);
		
		return _attBuilder.build();
		
	}
	
	public static  AttributeProto padsToAttributes(int[] pads)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("pads");
		_attBuilder.setType(AttributeType.INTS);
		
		if(pads== null)
		{
			pads = new int[] {0,0,0,0};
		}
		
		for(int i=0;i<pads.length;i++)
           _attBuilder.addInts(pads[i]);
		
		return _attBuilder.build();
		
	}
	
	public static AttributeProto kernelShapeToAttributes(int kernelSize)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("kernel_shape");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(kernelSize);
        _attBuilder.addInts(kernelSize);
        
		return _attBuilder.build();
		
	}
	
	
	public static AttributeProto kernelShapeToAttributes1D(int kernelSize)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("kernel_shape");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(kernelSize);
        _attBuilder.addInts(1);
        
		return _attBuilder.build();
		
	}
	
	public static AttributeProto dilationShapeToAttributes(int dilation)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("dilations");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(dilation);
        _attBuilder.addInts(dilation);
        
		return _attBuilder.build();
		
	}
	public static AttributeProto strideToAttributes1D(int stride)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("strides");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(stride);
        _attBuilder.addInts(1);
        
		return _attBuilder.build();
		
	}
	
	public static AttributeProto strideToAttributes(int stride)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("strides");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(stride);
        _attBuilder.addInts(stride);
        
		return _attBuilder.build();
		
	}
	
	public static AttributeProto strideToAttributes(int strideW, int strideH )
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("strides");
		_attBuilder.setType(AttributeType.INTS);
		
		//Add twice - From our format to onnx
        _attBuilder.addInts(strideW);
        _attBuilder.addInts(strideH);
        
		return _attBuilder.build();
		
	}
	
	public static AttributeProto groupToAttributes(int group)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("group");
		_attBuilder.setType(AttributeType.INT);
		
		//Add twice - From our format to onnx
        _attBuilder.setI(group);
        
		return _attBuilder.build();
		
	}
	
	private static AttributeProto boundaryToAttributes(BoundaryMode bm)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName("auto_pad");
		_attBuilder.setType(AttributeType.STRING);
		
		//Boundary mode is set as auto_pad in onnx. in our format there is only SAME - setting it to SAME_UPPER in onnx for now. 
		if(bm.equals(BoundaryMode.VALID))
			_attBuilder.setS(ByteString.copyFromUtf8("VALID"));
		else
			_attBuilder.setS(ByteString.copyFromUtf8("SAME"));
			
		return _attBuilder.build();
		
	}

	public static AttributeProto intValueToAttribute(String attributeName, int value)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName(attributeName);
		_attBuilder.setType(AttributeType.INT);
		_attBuilder.setI(value);
		return _attBuilder.build();
		
	}
	
	public static AttributeProto floatValueToAttribute(String attributeName, float value)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName(attributeName);
		_attBuilder.setType(AttributeType.FLOAT);
		_attBuilder.setF(value);
		return _attBuilder.build();
		
	}
	
	@SuppressWarnings("unused")
	private AttributeProto StringValueToAttribute(String attributeName, String text)
	{
		AttributeProto.Builder _attBuilder = AttributeProto.newBuilder();
		_attBuilder.setName(attributeName);
		_attBuilder.setType(AttributeType.STRING);
		_attBuilder.setS(ByteString.copyFromUtf8(text));
		return _attBuilder.build();
		
	}
	
	
}



/** NOTES
 *  
 *  
 * 
 * */
