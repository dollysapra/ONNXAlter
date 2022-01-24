package nl.uva.aloha.cells;

import java.util.ArrayList;
import java.util.HashMap;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import io.jenetics.util.RandomRegistry;
import nl.uva.aloha.cells.OperatorEdge.OperatorType;
import nl.uva.aloha.cells.OperatorNode.OperatorNodeType;
import nl.uva.aloha.converters.OnnxNodeInfoHolder;


public class Cell 
{
	
	public class Connection{
		OperatorNode input;
		OperatorNode output;
		public Connection(OperatorNode input, OperatorNode output) {
			this.input = input; this.output = output;
		}
	}

	public enum CellType {
		NORMAL,
		REDUCE
	}
	public OperatorNode inputCur;
	public OperatorNode inputPrev;
	public OperatorNode concat;
	OperatorNode outputNode;
	public ArrayList<OperatorNode> intermeds;
	
	public OperatorNode output;
	// list of connexions for each preprocessor - <operatorEdge list>.size == intermeds.size
	public HashMap<OperatorNode, ArrayList<OperatorEdge>  > connections;
	
	public Tensor InputfeatureMapSize;
	public Tensor OutputfeatureMapSize; //intermedfeaturemapize = outputfeaturemapsize/ intermeds.size
	
	//public int insideNumNeurons;
	public int numIntermeds;
	
	private CellType celltype;
	
	private int _scalefactor = 3; 
	public int maxScaleFactor = 5;
	public int minScaleFactor = 2;
	public int baseNumNeurons;
	Tensor prevLayerFeatureMap;
	protected ArrayList<String> _inputnames;
	
	public Cell(CellType ct, ArrayList<String> inputs, Tensor inputFeaturemap, int baseNumNeurons, Tensor prevLayerFeatureMap)
	{
		celltype = ct;
		
		InputfeatureMapSize = inputFeaturemap;
		OutputfeatureMapSize = inputFeaturemap.clone();
		if(ct == CellType.REDUCE)
		{
			OutputfeatureMapSize.setDimSize(0, inputFeaturemap.getDimSize(0)/2);
			OutputfeatureMapSize.setDimSize(1, inputFeaturemap.getDimSize(1)/2);
			
		}
		OutputfeatureMapSize.setDimSize(2, baseNumNeurons);
		
		this.baseNumNeurons = baseNumNeurons;
		this. prevLayerFeatureMap =  prevLayerFeatureMap;
		_inputnames = inputs;
		
		_scalefactor = RandomRegistry.getRandom().nextInt(maxScaleFactor - minScaleFactor) + minScaleFactor;
		createNew();
	}
	
	private void createNew()
	{
		intermeds = new ArrayList<OperatorNode>(_scalefactor);
		for(int i=0; i< _scalefactor; i++)
			intermeds.add(new OperatorNode(OperatorNodeType.ADD));
		
		
		if(InputfeatureMapSize.getDimSize(2) == baseNumNeurons)
			inputCur = new OperatorNode(OperatorNodeType.INPUT);
		else
		{
			inputCur = new OperatorNode(OperatorNodeType.INPUTPROCESS);
			inputCur.skipInFMapsForProcessor = InputfeatureMapSize.getDimSize(2) ;
		}
		
		
		inputCur.setName(_inputnames.get(0));
		
		
		if(prevLayerFeatureMap.getDimSize(0) != InputfeatureMapSize.getDimSize(0))
		{
			inputPrev = new OperatorNode(OperatorNodeType.INPUTPROCESSREDUCE);
			inputPrev.skipInFMapsForProcessor = prevLayerFeatureMap.getDimSize(2);
		}
		else if(prevLayerFeatureMap.getDimSize(2) != baseNumNeurons)
		{
			inputPrev = new OperatorNode(OperatorNodeType.INPUTPROCESS);
			inputPrev.skipInFMapsForProcessor = prevLayerFeatureMap.getDimSize(2);
		}
		else
			inputPrev = new OperatorNode(OperatorNodeType.INPUT);
		
		inputPrev.setName(_inputnames.get(1));
		output = new OperatorNode(OperatorNodeType.OUTPUT);
		concat = new OperatorNode(OperatorNodeType.CONCAT);
		setupInputNodes();
		setRandomNewConnectionsToIntermeds();
		setIntermedsToConcatConnections();
		setConcatToOutput();
	}
	
	private void setupInputNodes()
	{
		inputCur.setInputOutputFormat(InputfeatureMapSize);
		inputPrev.setInputOutputFormat(InputfeatureMapSize);
	}
	
	private void setRandomNewConnectionsToIntermeds()
	{
		//ArrayList<Connection> connections = new ArrayList<Connection>();
		ArrayList<OperatorNode> possibleInputs = new ArrayList<>();
		possibleInputs.add(inputCur);
		possibleInputs.add(inputPrev);
		if(celltype == CellType.NORMAL)
			possibleInputs.addAll(intermeds);
		
		for(int i=0; i< _scalefactor; i++)
		{
			
			ArrayList<OperatorEdge> tempNodeInp = new ArrayList<OperatorEdge>();
			
			int first = RandomRegistry.getRandom().nextInt(2);
			int other =  RandomRegistry.getRandom().nextInt(possibleInputs.size());
			
			tempNodeInp.add(new OperatorEdge(celltype, OperatorType.CHOOSE, possibleInputs.get(first), intermeds.get(i), InputfeatureMapSize,baseNumNeurons));
			
			
			while(true)
			{
				Boolean cyclic = isNewConnectionCyclic(possibleInputs.get(other), intermeds.get(i));
				if((possibleInputs.get(other).equals(intermeds.get(i))) || (cyclic))
					other =  RandomRegistry.getRandom().nextInt(possibleInputs.size());
				else
					break;
			}
			tempNodeInp.add(new OperatorEdge(celltype, OperatorType.CHOOSE, possibleInputs.get(other),intermeds.get(i), InputfeatureMapSize, baseNumNeurons));
			
			intermeds.get(i).setInputs(tempNodeInp);
		}
	}
	
	private Boolean isNewConnectionCyclic(OperatorNode in, OperatorNode out)
	{
		Boolean cyclic = false;
		if(in.getInputs() == null)
			return false;
		
		
		for (OperatorEdge op : in.getInputs()) 
		{
	        if (op.input.equals(out)) 
	        {
	        	cyclic =  true;
	        }
	        else cyclic = isNewConnectionCyclic(op.input, out);
	        
		}
		System.out.println("in:"+ in.name + " out:"+out.name + " cyclic:"+cyclic.toString());
		return cyclic;
	}
	
	private void setIntermedsToConcatConnections()
	{
		ArrayList<OperatorEdge> concatInputs = new ArrayList<>();
		for(int i=0; i< _scalefactor; i++)
		{
			concatInputs.add(new OperatorEdge(CellType.NORMAL, OperatorType.SAME, intermeds.get(i), concat,InputfeatureMapSize, baseNumNeurons));
		}
		
		concat.setInputs(concatInputs);
		
	}
	private void setConcatToOutput()
	{
		OperatorEdge outputProcessor = new OperatorEdge(CellType.NORMAL, OperatorType.SKIP, concat, output, InputfeatureMapSize, baseNumNeurons);
		ArrayList<OperatorEdge> concat2op = new ArrayList<>(); 
		concat2op.add(outputProcessor);
		output.setInputs( concat2op);	
	}
	
	public void mutate()
	{
		int randomIndex = RandomRegistry.getRandom().nextInt(intermeds.size());
		OperatorNode intermedToMutate = intermeds.get(randomIndex);
		
		randomIndex = RandomRegistry.getRandom().nextInt(2);
		OperatorEdge edgeTomutate = intermedToMutate.getInputs().get(randomIndex);
		edgeTomutate.changeOperatorTypeTo(OperatorType.CHOOSE);
	}
	
	private static ArrayList<OperatorNode> addednodes;// = new ArrayList<>();
	
	public ArrayList<OnnxNodeInfoHolder> getOnnxOps()
	{
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
		
		if( inputPrev.getOpType() != OperatorNodeType.INPUT) 
		{
			ops.addAll(inputPrev.getOpInfoFromOperatorNode());
		}
		if( inputCur.getOpType() != OperatorNodeType.INPUT)
		{
			ops.addAll(inputCur.getOpInfoFromOperatorNode());
			
		}
		addednodes = new ArrayList<>();
		addednodes.add(inputPrev);
		addednodes.add(inputCur);
		
		for(int i=0; i< intermeds.size(); i++)
		{
			OperatorNode node = intermeds.get(i);
			ops.addAll(addnoderecursily(node));
		}
		
		
		ops.addAll(concat.getOpInfoFromOperatorNode());
		ops.addAll(output.getOpInfoFromOperatorNode());
		return ops;
	}
	
	public ArrayList<OnnxNodeInfoHolder> getOnnxOpsCopy(String namePrefix)
	{
		ArrayList<OnnxNodeInfoHolder> ops = getOnnxOps();
		ArrayList<OnnxNodeInfoHolder> newOps = new ArrayList<>();
		
		for(int i=0; i< ops.size(); i++)
		{
			newOps.add(ops.get(i).getCopy(namePrefix));
		}
		
		
		return ops;
	}
	
	private ArrayList<OnnxNodeInfoHolder> addnoderecursily(OperatorNode node)
	{
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
		for(int j=0; j< node.getInputs().size(); j++)
		{
			if(!(addednodes.contains(node.getInputs().get(j).input)))
			{
				System.out.println("in"+node.getInputs().get(j).input.name);
				//addednodes.add(node.getInputs().get(j).input);
				ops.addAll(addnoderecursily(node.getInputs().get(j).input));
				
				
			}
		}
		
		if(!(addednodes.contains(node)))
		{
			System.out.println("out"+node.name);
			ops.addAll(node.getOpInfoFromOperatorNode());
			addednodes.add(node);
		}
		
		return ops; 
		
	}
	
//	
//	public static void main(final String[] args)
//	{
//		ArrayList<String> inputs = new ArrayList<>();
//		inputs.add("input_data");
//		inputs.add("input_data");
//	
//		Cell trial = new Cell(CellType.NORMAL, inputs, new Tensor(32,32), 16);
//		CellsToOnnx converter = new CellsToOnnx(trial);
//		OnnxHelper.saveOnnx(converter.convertToONNXModel());
//		
//		trial.mutate();
//		OnnxHelper.saveOnnx(converter.convertToONNXModel());
//		
//		trial.mutate();
//		OnnxHelper.saveOnnx(converter.convertToONNXModel());
//		
//		System.out.println("done");
//	}
}
