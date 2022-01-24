package nl.uva.aloha.cells;

import java.util.ArrayList;
import java.util.UUID;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import io.jenetics.util.RandomRegistry;
import nl.uva.aloha.converters.OnnxNodeInfoHolder;
import nl.uva.aloha.converters.OnnxNodeInfoHolder.OnnxOperatorTypes;

public class OperatorEdge 
{
	public enum OperatorType {
		  SAME,
		  SEP3x3,
		  SEP5x5,
		  SEPD3x3,
		  SEPD5x5,
		 // ZERO,
		  AVG_POOL2x2,
		  MAX_POOL2x2,
		  AVG_POOL3x3,
		  MAX_POOL3x3,
		  CHOOSE,
		  SKIP,
		}
	
	
	public OperatorType opType;
	public Cell.CellType cellType;
	protected String name; //optional
	
	public OperatorNode input;
	public OperatorNode output;
	
	protected Tensor _inputFormat;
	public Tensor outputFormat;
	
	private int _kernel;
	private int _pad;
	public int halfpad = 0;
	private int _stride;
	private int _dilation;
	public int numNeurons;
	
	public OperatorEdge(Cell.CellType ct, OperatorType op, OperatorNode input, OperatorNode output,Tensor inputFormat, int numNeurons)
	{
		this.input = input;
		this.output = output;
		this.numNeurons = numNeurons;
		cellType = ct;
		
		if(ct == Cell.CellType.NORMAL)
		{
			if(op == OperatorType.CHOOSE)
				opType = OperatorType.values()[RandomRegistry.getRandom().nextInt(OperatorType.values().length -2)]; // avoid choose and skip
			else
				opType = op;
			
			_stride =1;
		}
		if(ct == Cell.CellType.REDUCE)
		{
			if(op == OperatorType.CHOOSE)
				opType = OperatorType.values()[RandomRegistry.getRandom().nextInt(OperatorType.values().length -3) +1]; // avoid choose and skip
			else
				opType = op;
			
			_stride =2;
		}
		
		switch (opType)
		{
		
		case SKIP: _kernel = 1; _pad = 0;  _dilation = 1;
					name = "skip"+UUID.randomUUID().toString().replace("-", "");
			break;
		case SEP3x3: _kernel = 3; _pad = 1;  _dilation = 1;
			name = "sep3_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case SEP5x5: _kernel = 5; _pad = 2;  _dilation = 1;
			name = "sep5_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case SEPD3x3: _kernel = 3; _pad = 2;  _dilation = 2;
			name = "sepd3_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case SEPD5x5: _kernel = 5; _pad = 4;  _dilation = 2;
			name = "sepd5_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case AVG_POOL2x2: _kernel = 2;   _dilation = 1;
			name = "avgp2_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case MAX_POOL2x2:  _kernel = 2;   _dilation = 1;
			name = "maxp2_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case AVG_POOL3x3: _kernel = 3;  _dilation = 1;
			name = "avgp3_"+UUID.randomUUID().toString().replace("-", "");
			break;
		case MAX_POOL3x3:  _kernel = 3;  _dilation = 1;
			name = "maxp3_"+UUID.randomUUID().toString().replace("-", "");
			break;
		default: //identity & zero
			break;
		}
		setInputFormat(inputFormat);
	}

	public void setInputFormat(Tensor inputFormat)
	{
		_inputFormat = inputFormat;
		outputFormat = inputFormat.clone();
		if(cellType == Cell.CellType.REDUCE)
		{
			outputFormat.setDimSize(0, inputFormat.getDimSize(0)/2);
			outputFormat.setDimSize(0, inputFormat.getDimSize(0)/2);
		}
		switch (opType)
		{
		 case AVG_POOL3x3:
		 case MAX_POOL3x3: 
		 case AVG_POOL2x2:
		 case MAX_POOL2x2:
		 case SEP3x3:
		 case SEPD3x3:
		 case SEP5x5:
		 case SEPD5x5:
			 int w = inputFormat.getDimSize(0);
			 //int h = inputFormat.getDimSize(1);
			 int neww = outputFormat.getDimSize(0);
			 //int newh = outputFormat.getDimSize(1);
			 
			 int effectiveK = _dilation*(_kernel-1) + 1;
			 
			 //Assuming h & w are same - square feature map size;
			 int twiceP = (neww - 1)*_stride - w + effectiveK; 
			 
			 halfpad = (twiceP % 2 );
			 _pad = Math.floorDiv(twiceP, 2);
			 
			 //int neww = ((w-effectiveK + 2*_pad)/_stride) + 1;
			 //int newh = ((h-effectiveK + 2*_pad)/_stride) + 1;
			 break;
		default:
			_pad = 0;
			halfpad = 0;
			break;
				
		}
	}
	
	public ArrayList<OnnxNodeInfoHolder> getOpInfoFromOperatorEdge()
	{
		ArrayList<OnnxNodeInfoHolder> ops = new ArrayList<>();
		
		switch(opType)
		{
		case MAX_POOL2x2:
		case MAX_POOL3x3:
			OnnxNodeInfoHolder op = new OnnxNodeInfoHolder();
			op.name = name;
			op.OpType = OnnxOperatorTypes.MAXPOOL;
			op.kernel = _kernel;
			op.pads = new int[]{_pad,_pad,_pad+halfpad, _pad+halfpad};
			op.stride = _stride;
			op.inputs = new ArrayList<>();
			op.inputs.add(input.getOutputName());
			op.output = getOutputName();
			ops.add(op);
			break;
		case AVG_POOL2x2:
		case AVG_POOL3x3:
			OnnxNodeInfoHolder opp = new OnnxNodeInfoHolder();
			opp.name = name;
			opp.OpType = OnnxOperatorTypes.AVGPOOL;
			opp.kernel = _kernel;
			opp.pads = new int[]{_pad,_pad,_pad+halfpad, _pad+halfpad};
			opp.stride = _stride;
			opp.inputs = new ArrayList<>();
			opp.inputs.add(input.getOutputName());
			opp.output = getOutputName();
			ops.add(opp);
			break;
		 case SEP3x3:
		 case SEPD3x3:
		 case SEP5x5:
		 case SEPD5x5:
			 OnnxNodeInfoHolder opConv = new OnnxNodeInfoHolder();
			 opConv.name = name+"_d";
			 opConv.OpType = OnnxOperatorTypes.CONVOLUTION;
			 opConv.kernel = _kernel;
			 opConv.pads = new int[]{_pad,_pad,_pad+halfpad, _pad+halfpad};
			 opConv.stride = _stride;
			 opConv.dilations= _dilation;
			 opConv.group = input.numFMaps;
			 opConv.weightFormat = new Tensor(numNeurons,1,_kernel,_kernel);
			 opConv.biasFormat = new Tensor(numNeurons);
			 opConv.inputs = new ArrayList<>();
			 opConv.inputs.add(input.getOutputName());
			 opConv.output = name+"sep";
			 ops.add(opConv);
			 
			 OnnxNodeInfoHolder opConvO = new OnnxNodeInfoHolder();
			 opConvO.name = name+"_o";
			 opConvO.OpType = OnnxOperatorTypes.CONVOLUTION;
			 opConvO.kernel = 1;
			 opConvO.pads = new int[]{0,0,0,0,};
			 opConvO.stride = 1;
			 opConvO.group = 1;
			 opConvO.dilations = 1; 
			 opConvO.weightFormat = new Tensor(numNeurons,numNeurons,1,1);
			 opConvO.biasFormat = new Tensor(numNeurons);
			 opConvO.inputs = new ArrayList<>();
			 opConvO.inputs.add(name+"sep");
			 opConvO.output = getOutputName();
			 ops.add(opConvO);
			break; 
		 case SKIP:
			 OnnxNodeInfoHolder opC = new OnnxNodeInfoHolder();
			 opC.name = name;
			 opC.OpType = OnnxOperatorTypes.CONVOLUTION;
			 opC.kernel = 1;
			 opC.pads = new int[]{0,0,0,0,};
			 opC.stride = 1;
			 opC.weightFormat = new Tensor(numNeurons,input.numFMaps,1,1);
			 opC.biasFormat = new Tensor(numNeurons);
			 opC.dilations=1;
			 opC.group=1;
			 opC.inputs = new ArrayList<>();
			 opC.inputs.add(input.getOutputName());
			 opC.output = getOutputName();
			 ops.add(opC);
			 break;
		 
		}
		
		return ops;
	}
	
	public void changeOperatorTypeTo(OperatorType newOpType)
	{
		if(newOpType == OperatorType.CHOOSE)
			opType = OperatorType.values()[RandomRegistry.getRandom().nextInt(OperatorType.values().length -2)]; // avoid choose and skip
		else
			opType = newOpType;
		
		setInputFormat(_inputFormat); //recalculate padding etc.
	}
	
	public String getOutputName()
	{
		if(opType == OperatorType.SAME)
			return input.getOutputName();
		else return name+"_out";
	}
}
