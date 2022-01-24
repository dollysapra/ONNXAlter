package nl.uva.aloha.converters;

import java.util.ArrayList;

import espam.datamodel.graph.csdf.datasctructures.Tensor;

public class OnnxNodeInfoHolder 
{
	public enum OnnxOperatorTypes{
		CONVOLUTION,
		RELU,
		MAXPOOL,
		AVGPOOL,
		BATCHNORM,
		DATAIO,
		ADD,
		CONCAT,
		GLOBALAVGPOOL,
		FLATTEN,
		SOFTMAX,
		FC
	}
	
	public String name;
	public OnnxOperatorTypes OpType;
	public ArrayList<String> inputs;
	public String output;
	public Tensor weightFormat;
	public Tensor biasFormat;
	public int[] pads;
	public int kernel;
	public int stride;
	public int dilations;
	public int group;
	
	public OnnxNodeInfoHolder getCopy(String prefix)
	{
		OnnxNodeInfoHolder copy = new OnnxNodeInfoHolder();
		copy.name = prefix + this.name;
		copy.OpType = this.OpType;
		copy.inputs = new ArrayList<>();
		inputs.forEach(inp ->  copy.inputs.add(prefix+inp));
		copy.output = this.output;
		copy.weightFormat = this.weightFormat.clone();
		copy.biasFormat = this.biasFormat.clone();
		copy.pads = this.pads.clone();
		copy.kernel = this.kernel;
		copy.stride = this.stride;
		copy.dilations = this.dilations;
		copy.group = this.group;
		
		return copy;
	}
}
