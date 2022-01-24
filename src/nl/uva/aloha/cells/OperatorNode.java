package nl.uva.aloha.cells;

import java.util.ArrayList;
import java.util.UUID;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import nl.uva.aloha.converters.OnnxNodeInfoHolder;
import nl.uva.aloha.converters.OnnxNodeInfoHolder.OnnxOperatorTypes;

public class OperatorNode 
{
	public Tensor outputFormat;
	//private Tensor _inputFormat;
	private ArrayList<OperatorEdge> _inputs;
	public ArrayList<OperatorEdge> output;
	
	protected String name; 
	public enum OperatorNodeType {
		ADD,
		CONCAT,
		INPUTPROCESS,
		INPUT,
		OUTPUT,
		INPUTPROCESSREDUCE
	}
	
	private OperatorNodeType opType;
	public Tensor _featureMapSize;
	public int numFMaps;
	public int skipInFMapsForProcessor;
	
	public OperatorNode(OperatorNodeType nodeType, ArrayList<OperatorEdge> inputs)
	{
		opType = nodeType;
		setInputs(inputs);
		setInitialNames();
	}
	
	public  OperatorNode(OperatorNodeType nodeType)
	{
		opType = nodeType;
		setInitialNames();
	}
	public OperatorNodeType getOpType()
	{
		return opType;
	}
	private void setInitialNames()
	{
		switch(opType)
		{
		case ADD: name = "add_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case CONCAT: name = "concat_" + UUID.randomUUID().toString().replace("-", "");
			break;
		case OUTPUT: name = "cellout_" + UUID.randomUUID().toString().replace("-", "");
			break;
		case INPUT: name = "cellin_" + UUID.randomUUID().toString().replace("-", "");
			break;
		}
	}
	
	public void setName(String name)
	{
		this.name = name;
	}
	public void setInputOutputFormat(Tensor featuremapSize)
	{
		this.numFMaps = featuremapSize.getDimSize(2) ;
		if( (opType == OperatorNodeType.INPUT) ||(opType == OperatorNodeType.OUTPUT))
		{
			_featureMapSize = featuremapSize.getSubTensor(0, 2);
			outputFormat = featuremapSize.clone();
			//outputFormat.addDimension(numFMaps);
		}
	}
	
	
	public void setInputs(ArrayList<OperatorEdge> inputs)
	{
		_inputs = inputs;
		_featureMapSize = inputs.get(0).outputFormat;//.getSubTensor(0, 2);
		int totalmaps =0;
		switch(opType)
		{
		case ADD: 
			totalmaps = _inputs.get(0).numNeurons;
			for(int i=1; i< _inputs.size(); i++)
				if(totalmaps != _inputs.get(i).numNeurons)
					System.err.print("ADD inputs are of different sizes");
			outputFormat = _featureMapSize.clone();
			outputFormat.addDimension(totalmaps);
			numFMaps = totalmaps;
			break;
			
		case CONCAT:
			totalmaps = 0;//inputs.get(0).numNeurons;
			for(int i=0; i< _inputs.size(); i++)
				totalmaps += _inputs.get(i).numNeurons;
			outputFormat = _featureMapSize.clone();
			outputFormat.addDimension(totalmaps);
			numFMaps = totalmaps;
			break;
		default:
			break;
		}
		
	}
	
	public ArrayList<OnnxNodeInfoHolder> getOpInfoFromOperatorNode()
	{
		
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
			
		if(getInputs() !=null)
		{
			for(int j=0; j< getInputs().size(); j++)
			{
				OperatorEdge edge = getInputs().get(j);
				ops.addAll(edge.getOpInfoFromOperatorEdge());
			}
		}
		switch(opType)
		{
		case ADD: 
			 OnnxNodeInfoHolder opA = new OnnxNodeInfoHolder();
			 opA.name = name;
			 opA.OpType = OnnxOperatorTypes.ADD;

			 opA.inputs = new ArrayList<>();
			 for(int i=0; i< _inputs.size(); i++)
			 {
				 opA.inputs.add(_inputs.get(i).getOutputName());
				
			 }
			 
			 opA.output = getOutputName();
			 ops.add(opA);
			break;
		case CONCAT:
			 OnnxNodeInfoHolder opC = new OnnxNodeInfoHolder();
			 opC.name = name;
			 opC.OpType = OnnxOperatorTypes.CONCAT;
			 opC.inputs = new ArrayList<>();
			 for(int i=0; i< _inputs.size(); i++)
			 {
				 opC.inputs.add(_inputs.get(i).input.getOutputName());
			 }
			 
			 opC.output = getOutputName();
			 ops.add(opC);
			break;
		case INPUTPROCESS:
			OnnxNodeInfoHolder opI = new OnnxNodeInfoHolder();
			
			 opI.name = name;
			 opI.OpType = OnnxOperatorTypes.CONVOLUTION;
			 opI.kernel = 1;
			 opI.pads = new int[]{0,0,0,0,};
			 opI.stride = 1;
			 opI.weightFormat = new Tensor(numFMaps,skipInFMapsForProcessor,1,1);
			 opI.biasFormat = new Tensor(numFMaps);
			 opI.dilations=1;
			 opI.group=1;
			 opI.inputs = new ArrayList<>();
			 opI.inputs.add(name);
			 opI.output = getOutputName();
			 ops.add(opI);
			break;
		case INPUTPROCESSREDUCE:
			OnnxNodeInfoHolder opM = new OnnxNodeInfoHolder();
			
			 opM.name = name;
			 opM.OpType = OnnxOperatorTypes.MAXPOOL;
			 opM.kernel = 2;
			 opM.pads = new int[]{0,0,0,0,};
			 opM.stride = 2;
			 opM.inputs = new ArrayList<>();
			 opM.inputs.add(name);
			 opM.output = getOutputName();
			 ops.add(opM);
			break;
			
		default:
			break;
		}
		
		return ops;
		
	}
	
	public ArrayList<OperatorEdge> getInputs()
	{
		return _inputs;
	}
	
	public String getOutputName()
	{
		if (opType == OperatorNodeType.INPUT)
			return name;
		else if ((opType == OperatorNodeType.INPUTPROCESS)||(opType == OperatorNodeType.INPUTPROCESSREDUCE))
			return name+"_ip";
		else if(opType == OperatorNodeType.OUTPUT)
			return _inputs.get(0).getOutputName();
		else return name+"_out";
	}
}
