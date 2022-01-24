package nl.uva.aloha.cells;

import java.util.ArrayList;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import nl.uva.aloha.cells.Cell.CellType;
import nl.uva.aloha.cells.OperatorNode.OperatorNodeType;
import nl.uva.aloha.converters.NodesToOnnx;
import nl.uva.aloha.converters.OnnxNodeInfoHolder;
import nl.uva.aloha.converters.OnnxNodeInfoHolder.OnnxOperatorTypes;
import onnx.ONNX.ModelProto;

public class StackedCellNetwork 
{
	protected OperatorNode inputToNetwork;
	protected OperatorNode inputProcessor;
	protected Tensor _inputfeatureMapSize;
	protected int _initialChannels = 3;
	
	public static String onnxpath = "PATH TO ONNX MODEL";
	public static ModelProto model = null;
	public StackedCellNetwork()
	{
		
		_inputfeatureMapSize = new Tensor(32,32,3);
		
		inputToNetwork = new OperatorNode(OperatorNodeType.INPUT);
		inputToNetwork.setName("input_data");
		inputProcessor = new OperatorNode(OperatorNodeType.INPUT);
		
		
		//OperatorEdge processor = new OperatorEdge(CellType.NORMAL, OperatorType.SEP3x3, inputToNetwork, inputProcessor, _inputfeatureMapSize,32 );
		 //ArrayList<OperatorEdge> inps = new ArrayList<OperatorEdge>();
		 //inps.add(processor);
		//inputProcessor.setInputs(inps);
		
		int channels = 16;
		OnnxNodeInfoHolder inp = getInputProcessor(channels);
		
		ArrayList<String> inputs = new ArrayList<>();
		inputs.add(inp.output);
		inputs.add("input_data");
		Cell cell11 = new Cell(CellType.NORMAL, inputs, new Tensor(32,32,channels), channels, _inputfeatureMapSize );
		
		inputs = new ArrayList<>();
		inputs.add(cell11.output.getOutputName());
		inputs.add(inp.output);
		Cell cell12 = new Cell(CellType.NORMAL, inputs, new Tensor(32,32,channels), channels, cell11.InputfeatureMapSize );
		
		inputs = new ArrayList<>();
		inputs.add(cell12.output.getOutputName());
		inputs.add(cell11.output.getOutputName());
		Cell cell13 = new Cell(CellType.NORMAL, inputs, new Tensor(32,32,channels), channels,  cell11.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell13.output.getOutputName());
		inputs.add(cell12.output.getOutputName());
		Cell cell2 = new Cell(CellType.REDUCE, inputs, new Tensor(32,32,channels), channels,cell12.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell2.output.getOutputName());
		inputs.add(cell13.output.getOutputName());
		Cell cell31 = new Cell(CellType.NORMAL, inputs, new Tensor(16,16,channels), channels, cell13.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell31.output.getOutputName());
		inputs.add(cell2.output.getOutputName());
		Cell cell32 = new Cell(CellType.NORMAL, inputs, new Tensor(16,16,channels), channels, cell2.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell32.output.getOutputName());
		inputs.add(cell31.output.getOutputName());
		Cell cell33 = new Cell(CellType.NORMAL, inputs, new Tensor(16,16,channels), channels, cell31.OutputfeatureMapSize);
		
		
		inputs = new ArrayList<>();
		inputs.add(cell33.output.getOutputName());
		inputs.add(cell32.output.getOutputName());
		Cell cell4 = new Cell(CellType.REDUCE, inputs, new Tensor(16,16,channels), channels, cell32.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell4.output.getOutputName());
		inputs.add(cell33.output.getOutputName());
		Cell cell5 = new Cell(CellType.NORMAL, inputs, new Tensor(8,8,channels), channels, cell33.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell5.output.getOutputName());
		inputs.add(cell4.output.getOutputName());
		Cell cell52 = new Cell(CellType.NORMAL, inputs, new Tensor(8,8,channels), channels, cell4.OutputfeatureMapSize);
		
		inputs = new ArrayList<>();
		inputs.add(cell52.output.getOutputName());
		inputs.add(cell5.output.getOutputName());
		Cell cell53 = new Cell(CellType.NORMAL, inputs, new Tensor(8,8,channels), channels, cell5.OutputfeatureMapSize);
		
		ArrayList<OnnxNodeInfoHolder> onnxOps = new ArrayList<>();
		onnxOps.add(inp);
		onnxOps.addAll(cell11.getOnnxOps());
		onnxOps.addAll(cell12.getOnnxOps());
		onnxOps.addAll(cell13.getOnnxOps());
		onnxOps.addAll(cell2.getOnnxOps());
		onnxOps.addAll(cell31.getOnnxOps());
		onnxOps.addAll(cell32.getOnnxOps());
		onnxOps.addAll(cell33.getOnnxOps());
		onnxOps.addAll(cell4.getOnnxOps());
		onnxOps.addAll(cell5.getOnnxOps());
		onnxOps.addAll(cell52.getOnnxOps());
		onnxOps.addAll(cell53.getOnnxOps());
		
		//onnxOps.addAll(getAuxOutputLayers(cell4.output.getOutputName(), cell4.baseNumNeurons, 1));
		//onnxOps.addAll(getAuxOutputLayers(cell2.output.getOutputName(), cell2.baseNumNeurons, 2));
		//onnxOps.addAll(getAuxOutputLayers(cell32.output.getOutputName(), cell32.baseNumNeurons, 3));
		//onnxOps.addAll(getAuxOutputLayers(cell52.output.getOutputName(), cell52.baseNumNeurons, 4));
		
		onnxOps.addAll(getOutputLayers(cell53.output.getOutputName(), cell53.baseNumNeurons));
		
		NodesToOnnx converter = new NodesToOnnx(onnxOps,true);
		
		model = converter.convertToONNXModel();
		//ONNXFileWorker.writeModel(converter.convertToONNXModel(),onnxpath);
		
	
	}

	private OnnxNodeInfoHolder getInputProcessor(int channels)
	{
		OnnxNodeInfoHolder opConv = new OnnxNodeInfoHolder();
		opConv.name = "input_conv";
		 opConv.OpType = OnnxOperatorTypes.CONVOLUTION;
		 opConv.kernel = 3;
		 opConv.pads = new int[]{1,1,1,1};
		 opConv.stride = 1;
		 opConv.dilations= 1;
		 opConv.group = 1;
		 opConv.weightFormat = new Tensor(channels,3,3,3);
		 opConv.biasFormat = new Tensor(channels);
		 opConv.inputs = new ArrayList<>();
		 opConv.inputs.add("input_data");
		 opConv.output = "input_conv_out";
		
		 return opConv;
		
	}
	
	private ArrayList<OnnxNodeInfoHolder> getOutputLayers(String input, int lastcellFmapSize)
	{
		
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
		OnnxNodeInfoHolder opGlobalAvgPool = new OnnxNodeInfoHolder();
		opGlobalAvgPool.name="gapool";
		opGlobalAvgPool.OpType = OnnxOperatorTypes.GLOBALAVGPOOL;
		opGlobalAvgPool.inputs = new ArrayList<>();
		opGlobalAvgPool.inputs.add(input);
		opGlobalAvgPool.output = "gapool_out";
		
		
		OnnxNodeInfoHolder opConv = new OnnxNodeInfoHolder();
		opConv.name = "output_conv";
		 opConv.OpType = OnnxOperatorTypes.CONVOLUTION;
		 opConv.kernel = 1;
		 opConv.pads = new int[]{0,0,0,0};
		 opConv.stride = 1;
		 opConv.dilations= 1;
		 opConv.group = 1;
		 opConv.weightFormat = new Tensor(10,lastcellFmapSize,1,1);
		 opConv.biasFormat = new Tensor(10);
		 opConv.inputs = new ArrayList<>();
		 opConv.inputs.add("gapool_out");
		 opConv.output = "output_conv_out";
		 
		 
		OnnxNodeInfoHolder opFlatten = new OnnxNodeInfoHolder();
		opFlatten.OpType = OnnxOperatorTypes.FLATTEN;
		opFlatten.name="flatten";
		opFlatten.inputs = new ArrayList<>();
		opFlatten.inputs.add("output_conv_out");
		opFlatten.output = ("flatten_out");
		
		OnnxNodeInfoHolder opSoftmax = new OnnxNodeInfoHolder();
		opSoftmax.OpType = OnnxOperatorTypes.SOFTMAX;
		opSoftmax.name="softmax";
		opSoftmax.inputs = new ArrayList<>();
		opSoftmax.inputs.add("flatten_out");
		opSoftmax.output = ("softmax_output");
		
		ops.add(opGlobalAvgPool);
		ops.add(opConv);
		ops.add(opFlatten);
		ops.add(opSoftmax);
		 return ops;
	}

	
	private ArrayList<OnnxNodeInfoHolder> getAuxOutputLayers(String input, int lastcellFmapSize, int id)
	{
		
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
		OnnxNodeInfoHolder opGlobalAvgPool = new OnnxNodeInfoHolder();
		opGlobalAvgPool.name="gapool"+id;
		opGlobalAvgPool.OpType = OnnxOperatorTypes.GLOBALAVGPOOL;
		opGlobalAvgPool.inputs = new ArrayList<>();
		opGlobalAvgPool.inputs.add(input);
		opGlobalAvgPool.output = "gapool" + id+ "_out";
		
		//int tempexpansion = 64;
		OnnxNodeInfoHolder opConv = new OnnxNodeInfoHolder();
		opConv.name = "output_conv"+id;
		 opConv.OpType = OnnxOperatorTypes.CONVOLUTION;
		 opConv.kernel = 1;
		 opConv.pads = new int[]{0,0,0,0};
		 opConv.stride = 1;
		 opConv.dilations= 1;
		 opConv.group = 1;
		 opConv.weightFormat = new Tensor(10,lastcellFmapSize,1,1);
		 opConv.biasFormat = new Tensor(10);
		 opConv.inputs = new ArrayList<>();
		 opConv.inputs.add("gapool" + id +"_out");
		 opConv.output = "output_conv" + id + "_out";
		 
		 
		OnnxNodeInfoHolder opFlatten = new OnnxNodeInfoHolder();
		opFlatten.OpType = OnnxOperatorTypes.FLATTEN;
		opFlatten.name="flatten"+id;
		opFlatten.inputs = new ArrayList<>();
		opFlatten.inputs.add("output_conv"+id+"_out");
		//opFlatten.inputs.add("gapool" + id +"_out");
		opFlatten.output = ("flatten" +id+"_out");
		
//		OnnxNodeInfoHolder opFC = new OnnxNodeInfoHolder();
//		opFC.OpType = OnnxOperatorTypes.FC;
//		opFC.name = "FC"+id;
//		 opFC.weightFormat = new Tensor(10,lastcellFmapSize);
//		 opFC.biasFormat = new Tensor(10);
//		 opFC.inputs = new ArrayList<>();
//		 opFC.inputs.add("flatten" +id+"_out");
//		 opFC.output = ("output_fc" +id+"_out");
		
		
		OnnxNodeInfoHolder opSoftmax = new OnnxNodeInfoHolder();
		opSoftmax.OpType = OnnxOperatorTypes.SOFTMAX;
		opSoftmax.name="softmax"+id;
		opSoftmax.inputs = new ArrayList<>();
		//opSoftmax.inputs.add("output_fc" + id + "_out");
		opSoftmax.inputs.add("flatten" +id+"_out");
		opSoftmax.output = ("softmax" + id + "_output");
		
		ops.add(opGlobalAvgPool);
		ops.add(opConv);
	
		ops.add(opFlatten);
		//ops.add(opFC);
		ops.add(opSoftmax);
		 return ops;
	}
	
	
}
