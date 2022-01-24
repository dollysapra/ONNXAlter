package nl.uva.aloha.converters;

import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import nl.uva.aloha.cells.Cell;
import nl.uva.aloha.converters.OnnxNodeInfoHolder.OnnxOperatorTypes;
import onnx.ONNX;
import onnx.ONNX.GraphProto;
import onnx.ONNX.ModelProto;
import onnx.ONNX.NodeProto;
import onnx.ONNX.OperatorSetIdProto;
import onnx.ONNX.ValueInfoProto;



/**
 * 
 * @author Dolly Sapra
 *
 * This class converts an array of nodes (OnnxNodeInfoHolder) to an ONNX.
 * 
 */


public class NodesToOnnx 
{

	final static Tensor INPUT_DATA_CIFAR_SHAPE = new Tensor(32,32,3,1);
	final static Tensor OUTPUT_DATA_CIFAR_SHAPE = new Tensor(10,1);
	private GraphProto _graphProto;
	ModelProto.Builder _modelBuilder;// = ModelProto.newBuilder();
	GraphProto.Builder _graphBuilder;// = GraphProto.newBuilder();
	public boolean BatchNormAfterConv = true;
	private Cell _cell;
	private ArrayList<OnnxNodeInfoHolder> _onnxOps;
	public NodesToOnnx(Cell cell)
	{
		_cell = cell;
	}
	
	public NodesToOnnx(ArrayList<OnnxNodeInfoHolder> onnxOps)
	{
		_onnxOps = onnxOps;
	}
	
	public NodesToOnnx(ArrayList<OnnxNodeInfoHolder> onnxOps, Boolean processConvs)
	{
		_onnxOps = onnxOps;
		if(processConvs == true)
			_onnxOps =	addBNReLu(_onnxOps);
		else
			_onnxOps = onnxOps;
	}
	
	public ModelProto convertToONNXModel()
	{
		_modelBuilder = ModelProto.newBuilder();
		_graphBuilder = GraphProto.newBuilder();
		
		ValueInfoProto valueProto = NetworkToOnnx.createInputProto("input_data", Tensor.reverse(INPUT_DATA_CIFAR_SHAPE));
		_graphBuilder.addInput(valueProto);
	
		ValueInfoProto valueProto_o = NetworkToOnnx.createInputProto("softmax_output", Tensor.reverse(OUTPUT_DATA_CIFAR_SHAPE));
		_graphBuilder.addOutput(valueProto_o);
		
	/*	ValueInfoProto valueProto1_o = GeneToOnnx.createInputProto("softmax1_output", Tensor.reverse(OUTPUT_DATA_CIFAR_SHAPE));
		_graphBuilder.addOutput(valueProto1_o);
		
		ValueInfoProto valueProto2_o = GeneToOnnx.createInputProto("softmax2_output", Tensor.reverse(OUTPUT_DATA_CIFAR_SHAPE));
		_graphBuilder.addOutput(valueProto2_o);*/
//		ValueInfoProto valueProto3_o = GeneToOnnx.createInputProto("softmax3_output", Tensor.reverse(OUTPUT_DATA_CIFAR_SHAPE));
//		_graphBuilder.addOutput(valueProto3_o);
//		ValueInfoProto valueProto4_o = GeneToOnnx.createInputProto("softmax4_output", Tensor.reverse(OUTPUT_DATA_CIFAR_SHAPE));
//		_graphBuilder.addOutput(valueProto4_o);
		
		Iterator<OnnxNodeInfoHolder> itr = null;
		
		if(_cell !=null)
			itr = _cell.getOnnxOps().iterator();
		else if(_onnxOps !=null)
			itr = _onnxOps.iterator();
		else
			System.err.println("ERROR Nothing to convert");
		
		while(itr.hasNext())
		{
			_graphBuilder.addNode(OpToOnnxNode(itr.next()));
		}
		_modelBuilder.setProducerName("ALOHA");
		_modelBuilder.addOpsetImport(setopImportSet());
		_modelBuilder.setIrVersion(ONNX.Version.IR_VERSION.getNumber());
		
		_graphBuilder.setName("GAtest"+new Date().getTime());
		_graphProto = _graphBuilder.build();
		_modelBuilder.setGraph(_graphProto);
		
		return(_modelBuilder.build());
	}
	
	
	public NodeProto OpToOnnxNode(OnnxNodeInfoHolder op)
	{
		NodeProto.Builder _nodeBuilder = NodeProto.newBuilder();
		
		_nodeBuilder.setName(op.name);
		
		for(int i =0; i < op.inputs.size(); i++)
		{ 
			_nodeBuilder.addInput(op.inputs.get(i));
		}
		_nodeBuilder.addOutput( op.output);
		
		
		if(op.OpType== OnnxOperatorTypes.CONVOLUTION)
		{
			_nodeBuilder.setOpType("Conv");
			_nodeBuilder.addAttribute(NetworkToOnnx.padsToAttributes(op.pads));
			_nodeBuilder.addAttribute(NetworkToOnnx.kernelShapeToAttributes(op.kernel));
			_nodeBuilder.addAttribute(NetworkToOnnx.strideToAttributes( op.stride));
			_nodeBuilder.addAttribute(NetworkToOnnx.groupToAttributes(op.group));
			_nodeBuilder.addAttribute(NetworkToOnnx.dilationShapeToAttributes(op.dilations));
			
			_nodeBuilder.addInput(op.name + "_weight");
			_nodeBuilder.addInput(op.name + "_bias");
			
			_graphBuilder.addInput(NetworkToOnnx.createInputProto( op.name + "_weight", op.weightFormat));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_bias", op.biasFormat) );
			
			_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(op.name + "_weight",  op.weightFormat));
			_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(op.name + "_bias",  op.biasFormat ));
			
		}
		
		else if(op.OpType== OnnxOperatorTypes.MAXPOOL)
		{
			_nodeBuilder.setOpType("MaxPool");
			_nodeBuilder.addAttribute(NetworkToOnnx.padsToAttributes(op.pads));
			_nodeBuilder.addAttribute(NetworkToOnnx.kernelShapeToAttributes(op.kernel));
			_nodeBuilder.addAttribute(NetworkToOnnx.strideToAttributes( op.stride));
		}
			
		if(op.OpType== OnnxOperatorTypes.AVGPOOL)
		{
			_nodeBuilder.setOpType("AveragePool");
			_nodeBuilder.addAttribute(NetworkToOnnx.padsToAttributes(op.pads));
			_nodeBuilder.addAttribute(NetworkToOnnx.kernelShapeToAttributes(op.kernel));
			_nodeBuilder.addAttribute(NetworkToOnnx.strideToAttributes( op.stride));
			
		}
		else if(op.OpType== OnnxOperatorTypes.RELU)
		{
			_nodeBuilder.setOpType("LeakyRelu");
		}
		
		else if(op.OpType== OnnxOperatorTypes.ADD)
		{
			_nodeBuilder.setOpType("Add");
		}
		else if(op.OpType== OnnxOperatorTypes.CONCAT)
		{
			_nodeBuilder.setOpType("Concat");
			_nodeBuilder.addAttribute(NetworkToOnnx.intValueToAttribute("axis", 1));
		}
		else if(op.OpType == OnnxOperatorTypes.BATCHNORM)
		{
			_nodeBuilder.setOpType("BatchNormalization");
			
			_nodeBuilder.addAttribute(NetworkToOnnx.floatValueToAttribute("epsilon", (float)0.00001));
			_nodeBuilder.addAttribute(NetworkToOnnx.floatValueToAttribute("momentum", (float)0.9));
			
			_nodeBuilder.addInput(op.name + "_gamma");
			_nodeBuilder.addInput(op.name + "_beta");
			_nodeBuilder.addInput(op.name + "_mean");
			_nodeBuilder.addInput(op.name + "_var");
			
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_gamma", op.biasFormat));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_beta", op.biasFormat));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_mean", op.biasFormat));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_var", op.biasFormat));
		   
			
			_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(op.name + "_gamma", op.biasFormat));
			_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(op.name + "_beta", op.biasFormat));
			_graphBuilder.addInitializer(NetworkToOnnx.createZeroWeights(op.name + "_mean", op.biasFormat));
			_graphBuilder.addInitializer(NetworkToOnnx.createOneValuedWeights(op.name + "_var", op.biasFormat));
			
		}
		
		else if(op.OpType== OnnxOperatorTypes.SOFTMAX)
		{
			_nodeBuilder.setOpType("Softmax");
		}
		else if(op.OpType== OnnxOperatorTypes.FLATTEN)
		{
			_nodeBuilder.setOpType("Flatten");
		}
		else if(op.OpType== OnnxOperatorTypes.GLOBALAVGPOOL)
		{
			_nodeBuilder.setOpType("GlobalAveragePool");
		}
		else if(op.OpType == OnnxOperatorTypes.FC)
		{
			_nodeBuilder.setOpType("Gemm");
			//_nodeBuilder.addAttribute(GeneToOnnx.floatValueToAttribute("alpha", new Float(1.0)));
			//_nodeBuilder.addAttribute(GeneToOnnx.floatValueToAttribute("beta", new Float(1.0)));
			_nodeBuilder.addAttribute(NetworkToOnnx.intValueToAttribute("transB", 1));
			

			_nodeBuilder.addInput(op.name + "_weight");
			_nodeBuilder.addInput(op.name + "_bias");
			
			_graphBuilder.addInput(NetworkToOnnx.createInputProto( op.name + "_weight", op.weightFormat));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(op.name + "_bias", op.biasFormat) );
			
			_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(op.name + "_weight",  op.weightFormat));
			_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(op.name + "_bias",  op.biasFormat ));
			
			
		}
		return _nodeBuilder.build();
	}
	
	
	public static ArrayList<OnnxNodeInfoHolder>  addBNReLu(ArrayList<OnnxNodeInfoHolder> onnxOps)
	{
		ArrayList<OnnxNodeInfoHolder> opList = onnxOps;
		int i =0;
		do
		{
			OnnxNodeInfoHolder curOp = opList.get(i);
			String finaloutput = curOp.output;
			if(curOp.OpType == OnnxOperatorTypes.CONVOLUTION)
			{
				curOp.output = curOp.output+ "_toBN";
				OnnxNodeInfoHolder BN = new OnnxNodeInfoHolder();
				BN.name = curOp.name + "_BN";
				BN.OpType = OnnxOperatorTypes.BATCHNORM;
				BN.biasFormat = curOp.biasFormat.clone();
				BN.inputs = new ArrayList<>();
				BN.inputs.add(curOp.output);
				BN.output = curOp.output+ "_toRelu";
				opList.add(++i, BN);
				
				
				OnnxNodeInfoHolder Relu = new OnnxNodeInfoHolder();
				Relu.name = curOp.name + "_Relu";
				Relu.OpType = OnnxOperatorTypes.RELU;
				Relu.inputs = new ArrayList<>();
				Relu.inputs.add(BN.output);
				Relu.output = finaloutput;
				opList.add(++i, Relu);
				
				
			}
			i++;
		}while(i<opList.size());
		return opList;
	}
	
	private OperatorSetIdProto setopImportSet()
	{
		OperatorSetIdProto.Builder _opSetBuilder = OperatorSetIdProto.newBuilder();
		_opSetBuilder.setDomain("");
		_opSetBuilder.setVersion(9);
		return _opSetBuilder.build();
	}
	
}
