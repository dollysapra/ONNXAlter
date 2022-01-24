package nl.uva.aloha.Alterers;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

import com.google.protobuf.ByteString;

import espam.datamodel.graph.cnn.BoundaryMode;
import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.csdf.datasctructures.Tensor;
import espam.utils.fileworker.ONNXFileWorker;
import io.jenetics.util.RandomRegistry;
import nl.uva.aloha.converters.NetworkToOnnx;
import onnx.ONNX;
import onnx.ONNX.AttributeProto;
import onnx.ONNX.AttributeProto.AttributeType;
import onnx.ONNX.GraphProto;
import onnx.ONNX.ModelProto;
import onnx.ONNX.NodeProto;
import onnx.ONNX.NodeProto.Builder;
import onnx.ONNX.TensorProto;
import onnx.ONNX.TensorProto.DataType;
import onnx.ONNX.TensorShapeProto;
import onnx.ONNX.TensorShapeProto.Dimension;
import onnx.ONNX.ValueInfoProto;

public class ONNXAlteration 
{
	private ModelProto model;
	//private ModelProto alteredModel;
	private ModelProto.Builder  _modelBuilder;
	private GraphProto.Builder _graphBuilder;
	
	String _onnxFilePath;
	
	public ONNXAlteration(String onnxFilePath) 
	{
		_onnxFilePath = onnxFilePath;
		model = ONNXFileWorker.readModel(onnxFilePath);
		_modelBuilder = model.toBuilder();
		_graphBuilder = _modelBuilder.getGraphBuilder();
	}
	
	public ModelProto getOnnxModel()
	{
		return _modelBuilder.build();
	}
	
	public void updateONNXFile()
	{
		ONNXFileWorker.writeModel( _modelBuilder.build(), _onnxFilePath);	
	}
	
	public void recombineFrom(String onnx1filepath, String onnx2filepath )
	{
		GraphProto.Builder firstGraphBuilder  =   ONNXFileWorker.readModel(onnx1filepath).getGraph().toBuilder();
		GraphProto.Builder secondGraphBuilder =  ONNXFileWorker.readModel(onnx2filepath).getGraph().toBuilder();
		
		Map<String, ONNX.TensorProto> firstInitializerMap = getInitializerMap(firstGraphBuilder);
		
		Map<String, ONNX.TensorProto> secondInitializerMap = getInitializerMap(secondGraphBuilder);
		
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		
		while(it.hasNext())
		{
			TensorProto.Builder tensorBuilder = it.next();
			String initializerName = tensorBuilder.getName().split("_0")[0];
			ONNX.TensorProto initializerFromParent = null;
														
			boolean first = firstInitializerMap.containsKey(initializerName);
			boolean second  = secondInitializerMap.containsKey(initializerName);
			
			if(first && !second)
			{
				initializerFromParent = firstInitializerMap.get(initializerName);
			}
			else if (second && !first)
			{
				initializerFromParent = secondInitializerMap.get(initializerName);
			}
			else if (first && second)		
			{
				//if(firstInitializerMap.get(initializerName).getDimsList().equals(tensorBuilder.getDimsList()) )
				
				initializerFromParent = firstInitializerMap.get(initializerName);
				if(secondInitializerMap.get(initializerName).getDimsList().equals(tensorBuilder.getDimsList()))
					initializerFromParent = secondInitializerMap.get(initializerName);
				
				else if((tensorBuilder.getDimsCount()>2) && (initializerFromParent.getDims(2)!=tensorBuilder.getDims(2)))
				{
					initializerFromParent = secondInitializerMap.get(initializerName);
				}
					
			}
		
			//If initializer not found in either parent then just continue to next initializer 
														//and leave randomly initiated initializer as it is.
			else
				continue;
			
			if(initializerFromParent.getDimsList().equals(tensorBuilder.getDimsList()))
			{
				if(initializerFromParent.hasRawData())
					tensorBuilder.setRawData(getRawDataFromFloatData(initializerFromParent));
			}
			else if((initializerName.contains("dense") && initializerName.contains("weights") &&  
					(getProductOfLongList(initializerFromParent.getDimsList())  ==  getProductOfLongList(tensorBuilder.getDimsList())) ))
			{
				tensorBuilder.setRawData(transpose2DrawData(initializerFromParent));
				
				
			}
			else
			{
				ONNX.TensorProto.Builder initializerFromParentBuilder = initializerFromParent.toBuilder();
				if(initializerFromParent.getDims(0)!=tensorBuilder.getDims(0))
				{
					updateInitializerWeightFirstDim(initializerFromParentBuilder,(int) (tensorBuilder.getDims(0) - initializerFromParent.getDims(0)));
				}
				if((tensorBuilder.getDimsCount()>1)&&(initializerFromParent.getDims(1)!=tensorBuilder.getDims(1)))
				{
					updateInitializerWeightSecondDim(initializerFromParentBuilder,(int) (tensorBuilder.getDims(1) - initializerFromParent.getDims(1)));
				}
				
				tensorBuilder.setRawData(getRawDataFromFloatData(initializerFromParentBuilder));
			}
		}
		
	}
	
	
	public void mutateFrom(String onnxfilepath)
	{
		GraphProto.Builder originalGraphBuilder  =   ONNXFileWorker.readModel(onnxfilepath).getGraph().toBuilder();
		Map<String, ONNX.TensorProto> originalInitializerMap = getInitializerMap(originalGraphBuilder);
		
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		List<Integer> indexesToRemove = null;
		
		while(it.hasNext())
		{
		try 
		{
			TensorProto.Builder tensorBuilder = it.next();
			String initializerName = tensorBuilder.getName().split("_0")[0];
			
			ONNX.TensorProto initializerFromParent = originalInitializerMap.get(initializerName);
			if(initializerFromParent == null)
				continue;
			
			
			//TODO: Below if-else can be a separate function as this functionality is shared with recombinewith function.
			if(initializerFromParent.getDimsList().equals(tensorBuilder.getDimsList()))
			{
				tensorBuilder.setRawData(getRawDataFromFloatData(initializerFromParent));
//				System.out.print("raw");
			}
			else if((initializerName.contains("dense") && initializerName.contains("weights") &&  
					(getProductOfLongList(initializerFromParent.getDimsList())  ==  getProductOfLongList(tensorBuilder.getDimsList())) ))
			{
				tensorBuilder.setRawData(transpose2DrawData(initializerFromParent));
			}
			else
			{
				ONNX.TensorProto.Builder initializerFromParentBuilder = initializerFromParent.toBuilder();
				if((tensorBuilder.getDimsCount()>1)&&(initializerFromParent.getDims(1)!=tensorBuilder.getDims(1)))
				{ 
					int changeinDim = (int) (tensorBuilder.getDims(1) - initializerFromParent.getDims(1));
					if (initializerName.contains("weight") && (changeinDim < 0) && (indexesToRemove!=null)) 
					{
						indexedPruneInitializerWeightSecondDim(initializerFromParentBuilder, indexesToRemove, changeinDim);
//						System.out.print("prune2dim");
					}
					else
					{
						updateInitializerWeightSecondDim(initializerFromParentBuilder,changeinDim);
					//indexesToRemove = null; //After feature maps of subsequent layer have been removed - we can delete these indexes.
//						System.out.print("update2dim");
					}
				}
				
				if(initializerFromParent.getDims(0)!=tensorBuilder.getDims(0))
				{
					int changeinDim = (int) (tensorBuilder.getDims(0) - initializerFromParent.getDims(0));
					if (initializerName.contains("weight") && (changeinDim < 0))
					{
						indexesToRemove = pruneInitializerWeightFirstDim(initializerFromParentBuilder, changeinDim);
//						System.out.print("prune1dim");
					}
					else if((indexesToRemove!=null) && (tensorBuilder.getDimsCount() == 1)&&(changeinDim < 0) )
					{
						prune1DInitializer(initializerFromParentBuilder, indexesToRemove);
//						System.out.print("prune1diminit");
					}
					else
					{
						updateInitializerWeightFirstDim(initializerFromParentBuilder,changeinDim);
//						System.out.print("update1dim");
					}	
				}
				
				if((tensorBuilder.getDimsCount()>3) 
						//&& (tensorBuilder.getDims(2) == tensorBuilder.getDims(3)) 
						&& (initializerFromParent.getDims(2)!=tensorBuilder.getDims(2)))
				{
					
					if(NetworkToOnnx.DATASET.equals("PAMAP2"))
					{
						updateInitializerWeightKernelDim1D(initializerFromParentBuilder,(int) (tensorBuilder.getDims(2)), (int)(initializerFromParent.getDims(2)));
					}
					else
						updateInitializerWeightKernelDim(initializerFromParentBuilder,(int) (tensorBuilder.getDims(2)), (int)(initializerFromParent.getDims(2)));
//					System.out.print("updatekernel");
					
				}
				//Above functions modify initializerFromParentBuilder itself - hence the need to update them to correct ones here - 
				tensorBuilder.setRawData(getRawDataFromFloatData(initializerFromParentBuilder));
			}
		}catch(Exception e)
		{
			System.err.println(e.getMessage());
			throw e;
		}
		
		}//end of while
	}
	
	
	public long countParams()
	{
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		long count =0;
		while(it.hasNext()){
			TensorProto.Builder tensorBuilder = it.next();
			count += tensorBuilder.getDimsList().stream().mapToLong(a->a.longValue()).reduce((a,b) ->a*b).getAsLong();
		}
		
		return count;
	}
	
	public long countParamsAssumeBN()
	{
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		long count =0;
		while(it.hasNext()){
			TensorProto.Builder tensorBuilder = it.next();
			count += tensorBuilder.getDimsList().stream().mapToLong(a->a.longValue()).reduce((a,b) ->a*b).getAsLong();
			
			if(tensorBuilder.getName().startsWith("conv"))
			{
				count += 4*(tensorBuilder.getDims(0));
			}
					
		}
		return count;
	}
	
	public void changeNeuronNumsOfLayer(String layerName, int changeInNeurons)
	{
		if(changeInNeurons == 0)
			return;
		
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		
		while(it.hasNext())
		{
			TensorProto.Builder tensorBuilder = it.next();
			String initializerName = tensorBuilder.getName();
			if(initializerName.indexOf(layerName) == -1)
				continue;

			//System.out.println("initliazer found - " + initializerName);
			if(initializerName.equals(layerName+"weights"))
			{
				updateInitializerWeightFirstDim(tensorBuilder, changeInNeurons);
				updateInputWeightFirstDim(initializerName, tensorBuilder.getDims(0)+ changeInNeurons);
			}
			else if(initializerName.equals(layerName+"bias"))
			{
				updateInitializerBias(tensorBuilder, changeInNeurons);
				updateInputBias(initializerName, tensorBuilder.getDims(0)+changeInNeurons);
			}
		}
	}
	
	public void changeinputDimensionOfLayer(String layerName, int changeInNeurons)
	{
		if(changeInNeurons == 0)
			return;
		
		List<TensorProto.Builder> tbList = _graphBuilder.getInitializerBuilderList();
		Iterator<TensorProto.Builder> it = tbList.iterator();
		
		while(it.hasNext())
		{
			TensorProto.Builder tensorBuilder = it.next();
			String initializerName = tensorBuilder.getName();
			if(initializerName.indexOf(layerName) == -1)
				continue;

			//System.out.println("initliazer found - " + initializerName);
			if(initializerName.equals(layerName+"weights"))
			{
				updateInitializerWeightSecondDim(tensorBuilder, changeInNeurons);
				updateInputWeightSecondDim(initializerName, tensorBuilder.getDims(1)+ changeInNeurons);
			}
		}
	}
	
	public long checkReshapeLayerDim()
	{
		Iterator<NodeProto> it= _graphBuilder.getNodeList().iterator();
		String shapeName = "";
		
		while(it.hasNext())
		{
			NodeProto nb = it.next();
			if(nb.getOpType().contains("Reshape"))
			{
				shapeName = nb.getInput(1);
				//System.out.println(shapeName);
				Iterator<TensorProto.Builder> it_init = _graphBuilder.getInitializerBuilderList().iterator();
				while(it_init.hasNext())
				{
					TensorProto.Builder tb = it_init.next();
					ByteString initializerData = getRawDataFromFloatData(tb);
					if(tb.getName().equals(shapeName))
					{
						ByteBuffer bb = ByteBuffer.allocate(initializerData.size());
						initializerData.copyTo(bb);
						bb.order(ByteOrder.LITTLE_ENDIAN);
						return bb.getLong(0);
						
					
					}
				}
		
			}	
		}
		return -100;
		
	}
	
	public void resetReshapeLayerDimForTesting()
	{
		Iterator<NodeProto> it= _graphBuilder.getNodeList().iterator();
		String shapeName = "";
		
		while(it.hasNext())
		{
			NodeProto nb = it.next();
			if(nb.getOpType().contains("Reshape"))
			{
				shapeName = nb.getInput(1);
				//System.out.println(shapeName);
				Iterator<TensorProto.Builder> it_init = _graphBuilder.getInitializerBuilderList().iterator();
				while(it_init.hasNext())
				{
					TensorProto.Builder tb = it_init.next();
					ByteString initializerData = getRawDataFromFloatData(tb);
					
					if(tb.getName().equals(shapeName))
					{
						ByteBuffer bb = ByteBuffer.allocate(initializerData.size());
						initializerData.copyTo(bb);
						bb.order(ByteOrder.LITTLE_ENDIAN);
						bb.putLong(0, new Long(-1));
						//bb.asLongBuffer().put(0, -1);
						
						
						bb.rewind();
						tb.setRawData(ByteString.copyFrom(bb, bb.capacity()));
					
					}
				}
		
			}	
		}
	}
	public void removeLayerType( String layerType)
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		Iterator<NodeProto.Builder> it2 = _graphBuilder.getNodeBuilderList().iterator();
		//String shapeName = "";
		String inputToReshape = "";
		String outputOfReshape = "";
		int index =0;
		ArrayList<Integer> reshapeIndexes = new ArrayList<Integer>();
		while(it.hasNext())
		{
			NodeProto.Builder nb = it.next();
			if(nb.getOpType().contains(layerType))
			{
				inputToReshape = nb.getInput(0);
				outputOfReshape = nb.getOutput(0);
				reshapeIndexes.add(index);
			
				it2 = _graphBuilder.getNodeBuilderList().iterator();
				while(it2.hasNext())
				{
					NodeProto.Builder nb2 = it2.next();
					if(nb2.getInput(0).equals(outputOfReshape))
					{
						
						if(nb2.getOpType().contains("Matmul"))
						{
							nb2.setInput(0, nb2.getInput(1));
							nb2.setInput(1, inputToReshape);
						}
						else
						{
							nb2.setInput(0, inputToReshape);
						}
					}
				}
			
			
			}
			index++;
		}
		
		if(reshapeIndexes.size() > 0)
		{
			for(int i=reshapeIndexes.size()-1; i>=0; i--)
			{
				_graphBuilder.removeNode(reshapeIndexes.get(i));
			}
		}
		
	}
	
	
	
	
	public void removeReshapeLayer()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		Iterator<NodeProto.Builder> it2 = _graphBuilder.getNodeBuilderList().iterator();
		//String shapeName = "";
		String inputToReshape = "";
		String outputOfReshape = "";
		int index =0;
		ArrayList<Integer> reshapeIndexes = new ArrayList();
		while(it.hasNext())
		{
			NodeProto.Builder nb = it.next();
			if(nb.getOpType().contains("Reshape"))
			{
				inputToReshape = nb.getInput(0);
				outputOfReshape = nb.getOutput(0);
				reshapeIndexes.add(index);
			
				it2 = _graphBuilder.getNodeBuilderList().iterator();
				while(it2.hasNext())
				{
					NodeProto.Builder nb2 = it2.next();
					if(nb2.getInput(0).equals(outputOfReshape))
					{
						nb2.setInput(0, inputToReshape);
					}
				}
			
			
			}
			index++;
		}
		
		if(reshapeIndexes.size() > 0)
		{
			for(int i=reshapeIndexes.size()-1; i>=0; i--)
			{
				_graphBuilder.removeNode(reshapeIndexes.get(i));
			}
		}
			
	}
	
	public void removePadLayers()
	{

		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		Iterator<NodeProto.Builder> it2 = _graphBuilder.getNodeBuilderList().iterator();
		//String shapeName = "";
		String inputToReshape = "";
		String outputOfReshape = "";
		int index =0;
		ArrayList<Integer> reshapeIndexes = new ArrayList<Integer>();
		String padExtras = null;
		while(it.hasNext())
		{
			NodeProto.Builder nb = it.next();
			if(nb.getOpType().contains("Pad"))
			{
				inputToReshape = nb.getInput(0);
				outputOfReshape = nb.getOutput(0);
				padExtras = nb.getInput(1);
				reshapeIndexes.add(index);
			
				it2 = _graphBuilder.getNodeBuilderList().iterator();
				while(it2.hasNext())
				{
					NodeProto.Builder nb2 = it2.next();
					if(nb2.getInput(0).equals(outputOfReshape))
					{
						nb2.setInput(0, inputToReshape);
						if(nb2.getOpType().contains("Conv"))
						{
							//int[] pads = {1,1,1,1};
							int[] pads = {2,2,2,2};
							nb2.addAttribute(NetworkToOnnx.padsToAttributes(pads));
						}
						
					}
				}
		
			}
			index++;
		}
		
		if(reshapeIndexes.size() > 0)
		{
			for(int i=reshapeIndexes.size()-1; i>=0; i--)
			{
				_graphBuilder.removeNode(reshapeIndexes.get(i));
			}
		}
		

		int initIndex=0;
		ArrayList<Integer> padIndexes = new ArrayList<Integer>();
		TensorProto.Builder tensorBuilder = null;
		Iterator<TensorProto.Builder> ittt = _graphBuilder.getInitializerBuilderList().iterator();
		while(ittt.hasNext())
		{
			tensorBuilder = ittt.next();
			if(tensorBuilder.getName().contains("Pad"))
			{
				padIndexes.add(initIndex);
			}
			initIndex++;
		}
	
		if(padIndexes.size() > 0)
		{
			for(int i=padIndexes.size()-1; i>=0; i--)
			{
				_graphBuilder.removeInitializer(padIndexes.get(i));
			}
		}
		
		
		
	}
	
	
	public void fixInput()
	{
		Iterator<ValueInfoProto.Builder> itpp = _graphBuilder.getInputBuilderList().iterator();
		while(itpp.hasNext())
		{
			ValueInfoProto.Builder tba = itpp.next();
			Dimension.Builder d = Dimension.newBuilder();
			d.setDimValue(1);
			
			tba.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder().setDim(0,d);
		}
	}
	
	public void fixOutput()
	{
		Iterator<ValueInfoProto.Builder> itpp = _graphBuilder.getOutputBuilderList().iterator();
		while(itpp.hasNext())
		{
			ValueInfoProto.Builder tba = itpp.next();
			Dimension.Builder d = Dimension.newBuilder();
			d.setDimValue(1);
			
			tba.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder().setDim(0,d);
		}
	}
	
	public void fixPadLayers()
	{

		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();

		while(it.hasNext())
		{
			NodeProto.Builder nb = it.next();
			if(nb.getOpType().contains("Pad"))
			{
				String inputToReshape = nb.getInput(1);
				nb.setInput(1, "Pad8_pads");
			}
		}
		
		
	}
	
	public void addFlattenLayerAfterGlobalPool()
	{

		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC = "";
		String outputofglob = "";
		Integer index=0;
		Integer ind = -1;
		
		NodeProto.Builder builder = NodeProto.newBuilder();
		
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Global"))
			{
				outputofglob = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputofglob))
					{
						
						builder.setOpType("Flatten");
						builder.setName("flat" );
						builder.addInput(outputofglob);
						inputToFC = "flat_output";
						builder.addOutput( inputToFC);
						nb2.setInput(0, inputToFC);
						ind = index+1;
						break;
					}
					
				}
			}
			index++;
		}
		if(ind >-1)
		_graphBuilder.addNode(ind, builder);
	
	}

	
	public void ReplaceFeatureExtractorFC()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		int index=0;
		int fcind =0;
		NodeProto.Builder _nodeBuilder = NodeProto.newBuilder();
		_nodeBuilder.setName("gavgpool");
		_nodeBuilder.setOpType("GlobalAveragePool");
		_nodeBuilder.addOutput("gappool_output");
		
		NodeProto.Builder nb = null;
		
		while(it.hasNext())
		{
			NodeProto.Builder nbprev = nb;
			nb = it.next();
			if(nb.getOpType().contains("Gemm"))
			{
				if(_nodeBuilder.getInputCount() == 0)
				{
						_nodeBuilder.addInput(nbprev.getInput(0));
						fcind = index;
				}
				
			}
			if(nb.getOutput(0).contains("softmax_output"))
			{
				nbprev.setInput(0, "gappool_output");
				

				Iterator<TensorProto.Builder> ittt = _graphBuilder.getInitializerBuilderList().iterator();
				while(ittt.hasNext())
				{
					TensorProto.Builder tensorBuilder = ittt.next();
					String initializerName = tensorBuilder.getName();
					

					//System.out.println("initliazer found - " + initializerName);
					if(initializerName.indexOf(nbprev.getInput(1)) != -1) //WEIGHT
					{
						updateInitializerWeightSecondDim(tensorBuilder, 32);
						updateInputWeightSecondDim(initializerName, tensorBuilder.getDims(1));
					}
					
				}
			
				
				
			}
			
			
			index++;
			
		}
		
		 _graphBuilder.addNode(fcind -1, _nodeBuilder);
		 _graphBuilder.removeNode(fcind);
		 _graphBuilder.removeNode(fcind);
		 _graphBuilder.removeNode(fcind);
	}

	public void increaseOutputLayerby(int c)
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		
		while(it.hasNext())
		{
			NodeProto.Builder nb = it.next();
			if(nb.getOpType().contains("Gemm"))
			{
				NodeProto.Builder nbNext = it.next();
				if(nbNext.getOpType().contains("Softmax"))
				{
					Iterator<TensorProto.Builder> ittt = _graphBuilder.getInitializerBuilderList().iterator();
					while(ittt.hasNext())
					{
						TensorProto.Builder tensorBuilder = ittt.next();
						String initializerName = tensorBuilder.getName();
						

						//System.out.println("initliazer found - " + initializerName);
						if(initializerName.indexOf(nb.getInput(1)) != -1) //WEIGHT
						{
							updateInitializerWeightFirstDim(tensorBuilder, c);
							updateInputWeightFirstDim(initializerName, tensorBuilder.getDims(0));
						}
						else if(initializerName.indexOf(nb.getInput(2)) != -1)//BIAS
						{
							updateInitializerBias(tensorBuilder, c);
							updateInputBias(initializerName, tensorBuilder.getDims(0));
						}
					}
				}
			}
		}
		
		Iterator<ValueInfoProto.Builder> itoutp = _graphBuilder.getOutputBuilderList().iterator();
		while(itoutp.hasNext())
		{
			ValueInfoProto.Builder vb = itoutp.next();
			if(vb.getName().contains("softmax_output"))
			{
				TensorShapeProto.Builder shapeBuilder = vb.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder();
				shapeBuilder.getDimBuilder(1).setDimValue(11);
				break;
			}
		}
	}
	
	public void addDropOutLayersBetweenFC()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC2 = "";
		String outputOfFC1 = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> dropouts = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Gemm"))
			{
				outputOfFC1 = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfFC1))
					{
						if(!nb2.getOpType().contains("Gemm"))
							break;
						NodeProto.Builder dropoutBuilder = NodeProto.newBuilder();
						dropoutBuilder.setOpType("Dropout");
						dropoutBuilder.setName("dropout" + index);
						dropoutBuilder.addInput( outputOfFC1);
						inputToFC2 = "dropout" + index + "_output";
						AttributeProto.Builder attrib = AttributeProto.newBuilder();
						attrib.setName("ratio");
						attrib.setF((float)0.2);
						attrib.setType(AttributeType.FLOAT);
						
						dropoutBuilder.addAttribute(0, attrib);
						dropoutBuilder.addOutput( inputToFC2);
						dropoutBuilder.addOutput( "mask" + index);
						nb2.setInput(0, inputToFC2);
						dropouts.put(index+1,dropoutBuilder);
						index++;
						break;
					}
					
				}
				outputOfFC1 = "";
			}
			index++;
		}
		for(Map.Entry<Integer, NodeProto.Builder> entry : dropouts.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}				
	}
	

	public void addDropOutLayersBetweenFCRelu()
	{

		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC2 = "";
		String outputOfFC1 = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> dropouts = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Gemm"))
			{
				outputOfFC1 = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfFC1))
					{
						if(!nb2.getOpType().contains("Relu"))
							break;
						NodeProto.Builder dropoutBuilder = NodeProto.newBuilder();
						dropoutBuilder.setOpType("Dropout");
						dropoutBuilder.setName("dropout" + index);
						dropoutBuilder.addInput( outputOfFC1);
						inputToFC2 = "dropout" + index + "_output";
						AttributeProto.Builder attrib = AttributeProto.newBuilder();
						attrib.setName("ratio");
						attrib.setF((float)0.2);
						attrib.setType(AttributeType.FLOAT);
						
						dropoutBuilder.addAttribute(0, attrib);
						dropoutBuilder.addOutput( inputToFC2);
						dropoutBuilder.addOutput( "mask" + index);
						nb2.setInput(0, inputToFC2);
						dropouts.put(index+1,dropoutBuilder);
						index++;
						break;
					}
					
				}
				outputOfFC1 = "";
			}
			index++;
		}
		for(Map.Entry<Integer, NodeProto.Builder> entry : dropouts.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}				
	
	}
	
	public void addDropOutLayersBeforeFC()
	{


		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC2 = "";
		String outputOfFC1 = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> dropouts = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(true)
			{
				outputOfFC1 = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfFC1))
					{
						if(!nb2.getOpType().contains("Gemm"))
							break;
						NodeProto.Builder dropoutBuilder = NodeProto.newBuilder();
						dropoutBuilder.setOpType("Dropout");
						dropoutBuilder.setName("dropout" + index);
						dropoutBuilder.addInput( outputOfFC1);
						inputToFC2 = "dropout" + index + "_output";
						AttributeProto.Builder attrib = AttributeProto.newBuilder();
						attrib.setName("ratio");
						attrib.setF((float)0.2);
						attrib.setType(AttributeType.FLOAT);
						
						dropoutBuilder.addAttribute(0, attrib);
						dropoutBuilder.addOutput( inputToFC2);
						dropoutBuilder.addOutput( "mask" + index);
						nb2.setInput(0, inputToFC2);
						dropouts.put(index+1,dropoutBuilder);
						index++;
						break;
					}
					
				}
				outputOfFC1 = "";
			}
			index++;
		}
		for(Map.Entry<Integer, NodeProto.Builder> entry : dropouts.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}				
	
	
	}
	
	public void addDropOutLayersAfterBN()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC2 = "";
		String outputOfFC1 = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> dropouts = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Batch"))
			{
				outputOfFC1 = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfFC1))
					{
						//if(!nb2.getOpType().contains("Gemm"))
						//	break;
						NodeProto.Builder dropoutBuilder = NodeProto.newBuilder();
						dropoutBuilder.setOpType("Dropout");
						dropoutBuilder.setName("dropout" + index);
						dropoutBuilder.addInput( outputOfFC1);
						inputToFC2 = "dropout" + index + "_output";
						AttributeProto.Builder attrib = AttributeProto.newBuilder();
						attrib.setName("ratio");
						attrib.setF((float)0.2);
						attrib.setType(AttributeType.FLOAT);
						
						dropoutBuilder.addAttribute(0, attrib);
						dropoutBuilder.addOutput( inputToFC2);
						dropoutBuilder.addOutput( "mask" + index);
						nb2.setInput(0, inputToFC2);
						dropouts.put(index+1,dropoutBuilder);
						index++;
						break;
					}
					
				}
				outputOfFC1 = "";
			}
			index++;
		}
		for(Map.Entry<Integer, NodeProto.Builder> entry : dropouts.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}				
	}
	
	
	//Still to test. no idea if this works correctly.
	public void addDropOutLayersAfterBNReLu()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToFC2 = "";
		String outputOfFC1 = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> dropouts = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("ReLu"))
			{
				outputOfFC1 = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfFC1))
					{
						if(!nb2.getOpType().contains("Batch"))
							break;
						NodeProto.Builder dropoutBuilder = NodeProto.newBuilder();
						dropoutBuilder.setOpType("Dropout");
						dropoutBuilder.setName("dropout" + index);
						dropoutBuilder.addInput( outputOfFC1);
						inputToFC2 = "dropout" + index + "_output";
						AttributeProto.Builder attrib = AttributeProto.newBuilder();
						attrib.setName("ratio");
						attrib.setF((float)0.2);
						attrib.setType(AttributeType.FLOAT);
						
						dropoutBuilder.addAttribute(0, attrib);
						dropoutBuilder.addOutput( inputToFC2);
						dropoutBuilder.addOutput( "mask" + index);
						nb2.setInput(0, inputToFC2);
						dropouts.put(index+1,dropoutBuilder);
						index++;
						break;
					}
					
				}
				outputOfFC1 = "";
			}
			index++;
		}
		for(Map.Entry<Integer, NodeProto.Builder> entry : dropouts.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}				
	}
	
	public void resetInitializers()
	{
		Iterator<TensorProto> itInit= _graphBuilder.getInitializerList().iterator();
		int initIndex=0;
		while(itInit.hasNext())
		{
			TensorProto initializer = itInit.next();
			
			Tensor weightFormat = new Tensor();
			for(int i=0; i<initializer.getDimsCount(); i++)
				weightFormat.addDimension((int)initializer.getDims(i));
			
			_graphBuilder.setInitializer(initIndex, NetworkToOnnx.createHeWeights(initializer.getName(), weightFormat));
			initIndex++;
		}
	}
	
	
	public void traverseinitializers()
	{
		
		List<TensorProto.Builder> initMap = _graphBuilder.getInitializerBuilderList();
				//getInitializerMap(_graphBuilder);
		Iterator<TensorProto.Builder> it = initMap.iterator();
		while (it.hasNext())
		{
			
			TensorProto.Builder tp = it.next();
			String name = tp.getName();
			int dims = tp.getDimsCount();
			System.out.println(name + ":" + dims);
		}
		
	}
	public void traverseNodes()
	{
		
		List<Builder> initMap = _graphBuilder.getNodeBuilderList();
				//getInitializerMap(_graphBuilder);
		Iterator<Builder> it = initMap.iterator();
		while (it.hasNext())
		{
			
			Builder tp = it.next();
			String name = tp.getName();
			String type = tp.getOpType();
			System.out.println(name  + ":" + type);
		}
		
	}
	public void addBatchNormsAfterConv()
	{
		
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToNextRelu = "";
		String outputOfconv = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> BNs = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Conv"))
			{
				outputOfconv = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					
					
					if( (nb2.getInput(0).equals(outputOfconv)) || ( (nb2.getInputCount()>1) && (nb2.getInput(1).equals(outputOfconv))) )
					{
						//if(!nb2.getOpType().contains("Relu"))
						//	break;
						NodeProto.Builder BNBuilder = NodeProto.newBuilder();
						BNBuilder.setOpType("BatchNormalization");
						BNBuilder.setName("BN" + index);
						
						
						AttributeProto.Builder attrib1 = AttributeProto.newBuilder();
						attrib1.setName("epsilon");
						attrib1.setF((float)0.00001);
						attrib1.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib1);
						
						AttributeProto.Builder attrib2 = AttributeProto.newBuilder();
						attrib2.setName("momentum");
						attrib2.setF((float)0.9);
						attrib2.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib2);
						
						AttributeProto.Builder attrib3 = AttributeProto.newBuilder();
						attrib3.setName("spatial");
						attrib3.setI(0);
						attrib3.setType(AttributeType.INT);
						//BNBuilder.addAttribute(0, attrib3);
						
						BNBuilder.addInput(outputOfconv);
						String gamma = "BN" + index + "_gamma";
						String beta = "BN" + index + "_beta";
						String mean = "BN" + index + "_mean";
						String var = "BN" + index + "_var";
						
						BNBuilder.addInput(gamma);
						BNBuilder.addInput(beta);
						BNBuilder.addInput(mean);
						BNBuilder.addInput(var);
						
						
						Iterator<TensorProto> itl = _graphBuilder.getInitializerList().iterator();
						int ip =0;
						while(itl.hasNext())
						{
							TensorProto input = itl.next();
							if( input.getName().equals(nb1.getInput(2)))
								break;
							ip++;
						}
						
							
						long numNeurons = _graphBuilder.getInitializer(ip).getDimsList().get(0);
								//getType().getTensorType().getShape().getDim(0).getDimValue();
						Tensor format = new Tensor((int)numNeurons); //FIND FORMAT
						
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(gamma, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(beta, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(mean, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(var, format));
					   
						
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(gamma, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(beta, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createZeroWeights(mean, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createOneValuedWeights(var, format));
						
						inputToNextRelu = "BN" + index + "_output";
						BNBuilder.addOutput( inputToNextRelu);
						
						if(nb2.getInput(0).equals(outputOfconv))
						{
							nb2.setInput(0, inputToNextRelu);
						}
						else
						{
							nb2.setInput(1, inputToNextRelu);
						}
						BNs.put(index+1,BNBuilder);
						index++;
						break;
					}
					
				}
				outputOfconv = "";
			}
			index++;
		}
		
		for(Map.Entry<Integer, NodeProto.Builder> entry : BNs.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}		
	}
	
	public void convertConvToSeperable()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToNextRelu = "";
		String outputOfconv = "";
		Integer index=0;
		
		Map<Integer, NodeProto.Builder> BNs = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			if(nb1.getOpType().contains("Conv"))
			{
				outputOfconv = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfconv))
					{
						if(!nb2.getOpType().contains("Relu"))
							break;
						NodeProto.Builder BNBuilder = NodeProto.newBuilder();
						BNBuilder.setOpType("BatchNormalization");
						BNBuilder.setName("BN" + index);
						
						
						AttributeProto.Builder attrib1 = AttributeProto.newBuilder();
						attrib1.setName("epsilon");
						attrib1.setF((float)0.00001);
						attrib1.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib1);
						
						AttributeProto.Builder attrib2 = AttributeProto.newBuilder();
						attrib2.setName("momentum");
						attrib2.setF((float)0.9);
						attrib2.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib2);
						
						AttributeProto.Builder attrib3 = AttributeProto.newBuilder();
						attrib3.setName("spatial");
						attrib3.setI(0);
						attrib3.setType(AttributeType.INT);
						//BNBuilder.addAttribute(0, attrib3);
						
						BNBuilder.addInput(outputOfconv);
						String gamma = "BN" + index + "_gamma";
						String beta = "BN" + index + "_beta";
						String mean = "BN" + index + "_mean";
						String var = "BN" + index + "_var";
						
						BNBuilder.addInput(gamma);
						BNBuilder.addInput(beta);
						BNBuilder.addInput(mean);
						BNBuilder.addInput(var);
						
						
						Iterator<ValueInfoProto> itl = _graphBuilder.getInputList().iterator();
						int ip =0;
						while(itl.hasNext())
						{
							ValueInfoProto input = itl.next();
							if( input.getName().equals(nb1.getInput(2)))
								break;
							ip++;
						}
						
							
						long numNeurons = _graphBuilder.getInput(ip).getType().getTensorType().getShape().getDim(0).getDimValue();
						Tensor format = new Tensor((int)numNeurons); //FIND FORMAT
						
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(gamma, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(beta, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(mean, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(var, format));
					   
						
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(gamma, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(beta, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createZeroWeights(mean, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createOneValuedWeights(var, format));
						
						inputToNextRelu = "BN" + index + "_output";
						BNBuilder.addOutput( inputToNextRelu);
						
						nb2.setInput(0, inputToNextRelu);
						
						BNs.put(index+1,BNBuilder);
						index++;
						break;
					}
					
				}
				outputOfconv = "";
			}
			index++;
		}
		
		for(Map.Entry<Integer, NodeProto.Builder> entry : BNs.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}		
	
		
	}
	
	public void addBatchNormsAfterRelu()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		String inputToNextRelu = "";
		String outputOfconv = "";
		Integer index=0;
		long numNeurons =0; 
		
		Map<Integer, NodeProto.Builder> BNs = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			
			if(nb1.getOpType().contains("Conv"))
			{
				Iterator<ValueInfoProto> itl = _graphBuilder.getInputList().iterator();
				int ip =0;
				while(itl.hasNext())
				{
					ValueInfoProto input = itl.next();
					if( input.getName().equals(nb1.getInput(2)))
						break;
					ip++;
				}
				
					
				numNeurons = _graphBuilder.getInput(ip).getType().getTensorType().getShape().getDim(0).getDimValue();
			}
			
			else if(nb1.getOpType().contains("Relu"))
			{
				outputOfconv = nb1.getOutput(0);
				
				Iterator<NodeProto.Builder> it1 = _graphBuilder.getNodeBuilderList().iterator();
				while(it1.hasNext())
				{
					NodeProto.Builder nb2 = it1.next();
					if(nb2.getInput(0).equals(outputOfconv))
					{
						NodeProto.Builder BNBuilder = NodeProto.newBuilder();
						BNBuilder.setOpType("BatchNormalization");
						BNBuilder.setName("BN" + index);
						
						
						AttributeProto.Builder attrib1 = AttributeProto.newBuilder();
						attrib1.setName("epsilon");
						attrib1.setF((float)0.00001);
						attrib1.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib1);
						
						AttributeProto.Builder attrib2 = AttributeProto.newBuilder();
						attrib2.setName("momentum");
						attrib2.setF((float)0.9);
						attrib2.setType(AttributeType.FLOAT);
						BNBuilder.addAttribute(0, attrib2);
						
						AttributeProto.Builder attrib3 = AttributeProto.newBuilder();
						attrib3.setName("spatial");
						attrib3.setI(0);
						attrib3.setType(AttributeType.INT);
						//BNBuilder.addAttribute(0, attrib3);
						
						BNBuilder.addInput(outputOfconv);
						String gamma = "BN" + index + "_gamma";
						String beta = "BN" + index + "_beta";
						String mean = "BN" + index + "_mean";
						String var = "BN" + index + "_var";
						
						BNBuilder.addInput(gamma);
						BNBuilder.addInput(beta);
						BNBuilder.addInput(mean);
						BNBuilder.addInput(var);
						
						
						
						Tensor format = new Tensor((int)numNeurons); //FIND FORMAT
						
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(gamma, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(beta, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(mean, format));
						_graphBuilder.addInput(NetworkToOnnx.createInputProto(var, format));
					   
						
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(gamma, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(beta, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createZeroWeights(mean, format));
						_graphBuilder.addInitializer(NetworkToOnnx.createOneValuedWeights(var, format));
						
						inputToNextRelu = "BN" + index + "_output";
						BNBuilder.addOutput( inputToNextRelu);
						//BNBuilder.addOutput( mean);
						//BNBuilder.addOutput( var);
						//BNBuilder.addOutput( mean + "saved");
						//BNBuilder.addOutput( var + "saved");
						
						nb2.setInput(0, inputToNextRelu);
						
						BNs.put(index+1,BNBuilder);
						index++;
						numNeurons =0;
						break;
					}
					
				}
				outputOfconv = "";
			}
			index++;
		}
		
		for(Map.Entry<Integer, NodeProto.Builder> entry : BNs.entrySet()) 
		{
		    Integer ind = entry.getKey();
		    NodeProto.Builder dt = entry.getValue();

		    _graphBuilder.addNode(ind, dt);
		}		
	}
	
	public void correctpadding()
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		
		//Map<Integer, NodeProto.Builder> BNs = new TreeMap<>();
		while(it.hasNext())
		{
			NodeProto.Builder nb1 = it.next();
			
			if(nb1.getOpType().contains("Conv"))
			{
				int k = 0;
				int p =0;
				Iterator<AttributeProto.Builder> itA = nb1.getAttributeBuilderList().iterator();
				while(itA.hasNext())
				{
					AttributeProto.Builder atB = itA.next();
					
					if(atB.getName().contains("kernel_shape"))
					{
						k = (int)atB.getInts(0);
					}
					if(atB.getName().contains("pads"))
					{
						p = (int)atB.getInts(0);
					}
				}
				
				itA = nb1.getAttributeBuilderList().iterator();
				while(itA.hasNext())
				{
					AttributeProto.Builder atB = itA.next();
					int newp = Math.floorDiv(k, 2);
					if(newp==p)
						break;
					if(atB.getName().contains("pads"))
					{
						atB.setInts(0, newp);
						atB.setInts(1, newp);
						atB.setInts(2, newp);
						atB.setInts(3, newp);
					}
				}
				
			}
		}
		
	}
	public void renameInputOutputLayerAfterOneRun()
	{
		//Iterator<NodeProto> it= _graphBuilder.getNodeList().iterator();
		
		//ValueInfoProto.Builder 
		Iterator<ValueInfoProto.Builder> it= _graphBuilder.getInputBuilderList().iterator();
		while(it.hasNext())
		{
			ValueInfoProto.Builder vb = it.next();
			if(vb.getName().contains("input_data_0"))
				vb.setName("input_data");
		}
		
		it = _graphBuilder.getOutputBuilderList().iterator();
		while(it.hasNext())
		{
			ValueInfoProto.Builder vb = it.next();
			if(vb.getName().contains("softmax_output_1"))
				vb.setName("softmax_output");
		}
		
		Iterator<NodeProto.Builder> nodeIt= _graphBuilder.getNodeBuilderList().iterator();
		while(nodeIt.hasNext())
		{
			NodeProto.Builder nb = nodeIt.next();
			if(nb.getInput(0).contains("input_data_0"))
				nb.setInput(0, "input_data");
			else if(nb.getOutput(0).contains("softmax_output_1"))
				nb.setOutput(0, "softmax_output");
		}
	}
	
	public void addRandomSkipConnection()
	{
		ListIterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().listIterator(_graphBuilder.getNodeBuilderList().size());
		
		NodeProto.Builder addBuilder = NodeProto.newBuilder();
		int numReluLayers = 0;
		int selectedSecondLayer = -1;
		String addOutput = "addOutput";
		int addIndex = -1;
		String addInputOne = "";
		String addInputTwo = "";
		
		int numNeuronsOne = 0;
		int numNeuronsTwo = 0;
		
		boolean continueonce = true;
		
		while(it.hasPrevious())
		{
			NodeProto.Builder nb1 = it.previous();
			
			if(nb1.getOpType().contains("MaxPool"))
			{
				if(continueonce)
				{
					continueonce = false;
					continue;
				}
					
				
				addIndex = it.previousIndex();
				//nb1.setInput(0, addOutput);
				while(it.hasPrevious())
				{
					NodeProto.Builder nb2 = it.previous();
					if(nb2.getOpType().contains("Relu"))
					{
						numReluLayers++;
						
					}
						
					else if(nb2.getOpType().contains("MaxPool"))
					{
						selectedSecondLayer = RandomRegistry.getRandom().nextInt(numReluLayers);
						
						break;
					}
				}
				
				if(selectedSecondLayer > -1)
					break;
			}
			
		}
		
				
		if(selectedSecondLayer == 0)
			addInputTwo = it.next().getOutput(0); //it should be at maxpool now if selected layer >-1 (here it is zero)
		
		for(int i = 1; i<=numReluLayers; )
		{
			while(it.hasNext())
			{
					NodeProto.Builder nb1 = it.next();
					if(nb1.getOpType().contains("Relu"))
					{
						if(selectedSecondLayer == i)
							addInputTwo = nb1.getOutput(0);
						else if(numReluLayers == i) //Last Relu Layer
						{
							addInputOne = nb1.getInput(0);
							nb1.setInput(0, addOutput);
						}
						i++;
						break;
					}
						
			}
		}
		addBuilder.setOpType("Add");
		addBuilder.setName("add" + numReluLayers);
		
		
		addBuilder.addInput(addInputOne);
		addBuilder.addInput(addInputTwo);
		addBuilder.addOutput(addOutput);
		_graphBuilder.addNode(addIndex, addBuilder);
		
		
	}

	public void addChromosomeLevelSkipConnection()
	{
		int runningIndex = _graphBuilder.getNodeBuilderList().size();
		ListIterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().listIterator(runningIndex);
		
		int addIndex = -1;
		
		
		while(runningIndex > 1)
		{
			
			NodeProto.Builder nb1 = it.previous();
			NodeProto.Builder dim1Holder = null;
			if(nb1.getOpType().contains("MaxPool"))
			{
				addIndex = it.previousIndex(); //Insert "before" previous layer (before ReLu)
				NodeProto.Builder relu2 = it.previous();
				if(relu2.getOpType().contains("Relu")) //Confirm
				{
					//relu2.setInput(0, addOutput);
					dim1Holder = it.previous();
				}
				//Traverse till previous maxpool layer is reached
				while(it.hasPrevious())
				{
					runningIndex = it.nextIndex();
					NodeProto.Builder nb2 = it.previous();
					if(nb2.getOpType().contains("MaxPool"))
					{
						//maxpOutput = nb2.getOutput(0);
						it.next();//Refer back to maxpool
						NodeProto.Builder dim2Holder = it.next();
						if((dim1Holder !=null) &&(addIndex>0))
						{
							String skipOutput = addSkipConnectionToGraph(dim1Holder, dim2Holder, addIndex);
							relu2.setInput(0,skipOutput);
						}
						//runningIndex = 1;
						it= _graphBuilder.getNodeBuilderList().listIterator(runningIndex);
						break;
						
					}
					
				}//End of inner while loop.
			}//Maxpool condition
			
			
		}//End of outer while loop
		
	}
	
	
	
	private String addSkipConnectionToGraph(NodeProto.Builder dim1Holder, NodeProto.Builder dim2Holder, int addIndex)
	{
	
		int dim1 = 0;
		int dim2 = 0;

		String skipUID = UUID.randomUUID().toString().replace("-", "");
		
		Iterator<ValueInfoProto> itl = _graphBuilder.getInputList().iterator();

		while(itl.hasNext())
		{
			ValueInfoProto input = itl.next();
			if( input.getName().equals(dim1Holder.getInput(1)))
				dim1 = (int)input.getType().getTensorType().getShape().getDim(0).getDimValue();
			else if( input.getName().equals(dim2Holder.getInput(1)))
				dim2 = (int)input.getType().getTensorType().getShape().getDim(1).getDimValue();
			
		}
		
		Convolution cnv = new Convolution(1, BoundaryMode.VALID,1); //KernelSize, boundary, stride(default=1)
		Layer l = new Layer("convskip"+skipUID,cnv,dim1); // need to set numNeurons later
		l.setPads(0,0,0,0);
		//l.setInputFormat(new Tensor(dim1,dim2,1,1));
		
		String weightName = l.getName()+"weights";
		String biasName = l.getName()+"bias";
		ArrayList<String> convLayerInputs = new ArrayList<String>(); 
		ArrayList<String> convLayerOutputs = new ArrayList<String>(); 
		
		convLayerInputs.add(dim2Holder.getInput(0));
		convLayerInputs.add(weightName);
		convLayerInputs.add(biasName);
		convLayerOutputs.add(l.getName()+"_out");
		
		int k = 1;
		Tensor weightFormat = new Tensor(dim1,dim2,k,k);
				
		_graphBuilder.addInput(NetworkToOnnx.createInputProto( weightName,weightFormat));
		_graphBuilder.addInput(NetworkToOnnx.createInputProto(biasName, new Tensor(dim1) ));
		
		_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(weightName, weightFormat));
		_graphBuilder.addInitializer(NetworkToOnnx.createHeWeights(biasName, new Tensor(dim1) ));
		
		if(NetworkToOnnx.BatchNormAfterConv)
		{
			ArrayList<String> intermediateInputOutput = new ArrayList<String>(); 
			String intermediator = l.getName()+"interBN";
			String BNname = l.getName() + "BN";
			intermediateInputOutput.add(intermediator);
			
			_graphBuilder.addNode(addIndex++,NetworkToOnnx.layerToOnnxNode(l,convLayerInputs,intermediateInputOutput));
			
			NodeProto.Builder BNBuilder = NodeProto.newBuilder();
			BNBuilder.setOpType("BatchNormalization");
			BNBuilder.setName(BNname );
			BNBuilder.addAttribute(NetworkToOnnx.floatValueToAttribute("epsilon", (float)0.00001));
			BNBuilder.addAttribute(NetworkToOnnx.floatValueToAttribute("momentum", (float)0.9));
			
			BNBuilder.addInput(intermediator);
			BNBuilder.addInput(BNname + "_gamma");
			BNBuilder.addInput(BNname + "_beta");
			BNBuilder.addInput(BNname + "_mean");
			BNBuilder.addInput(BNname + "_var");
			
			Tensor format = new Tensor(l.getNeuronsNum());
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(BNname + "_gamma", format));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(BNname + "_beta", format));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(BNname + "_mean", format));
			_graphBuilder.addInput(NetworkToOnnx.createInputProto(BNname + "_var", format));
		   
			_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(BNname + "_gamma", format));
			_graphBuilder.addInitializer(NetworkToOnnx.createRandomWeight(BNname + "_beta", format));
			_graphBuilder.addInitializer(NetworkToOnnx.createZeroWeights(BNname + "_mean", format));
			_graphBuilder.addInitializer(NetworkToOnnx.createOneValuedWeights(BNname + "_var", format));
			
			BNBuilder.addOutput(convLayerOutputs.get(0));
			_graphBuilder.addNode(addIndex++,BNBuilder.build());
		}
		else
			_graphBuilder.addNode(addIndex++,NetworkToOnnx.layerToOnnxNode(l,convLayerInputs,convLayerOutputs));
		
		
		
		//NodeProto.Builder SkipConvBuilder = GeneToOnnx.layerToOnnxNode(l, inputs, outputs)
		//NodeProto.Builder BNBuilder = NodeProto.newBuilder();
		NodeProto.Builder addBuilder = NodeProto.newBuilder();
		
		addBuilder.setOpType("Add");
		addBuilder.setName("add"+skipUID );
		String addOutput = "add"+skipUID + "_out";
		
		addBuilder.addInput(convLayerOutputs.get(0));
		addBuilder.addInput(dim1Holder.getOutput(0));
		
		addBuilder.addOutput(addOutput);
		
		_graphBuilder.addNode(addIndex,addBuilder);
		
		return addOutput;
	}
	
	public void pruneBetter(int pc)
	{
		
	}
	
	public void pruneAll(int pc)
	{
		Iterator<NodeProto.Builder> it= _graphBuilder.getNodeBuilderList().iterator();
		//Iterator<TensorProto.Builder> it = tbList.iterator();
		
		while(it.hasNext())
		{
			NodeProto.Builder nodeBuilder = it.next();
			
			
			if ((nodeBuilder.getOpType().contains("Conv")) || (nodeBuilder.getOpType().contains("Gemm")))
			{
				String layerName = nodeBuilder.getName();
				if((layerName==null) || (layerName.length()==0))
				{
					nodeBuilder.setName(nodeBuilder.getInput(1).split("weights")[0]);
				}
					Iterator<TensorProto.Builder> ittb = _graphBuilder.getInitializerBuilderList().iterator();
					int changeinp = 0;
					int changelayer = 0;
					while(ittb.hasNext())
					{
						TensorProto.Builder tb = ittb.next();
						if(tb.getName().contains(nodeBuilder.getInput(1)))
						{
							changelayer = Math.round((pc* tb.getDims(0))/100);
							changeinp = Math.round((pc* tb.getDims(1))/100);
							break;
						}
						
					}
					if(changelayer > 1)
						changeNeuronNumsOfLayer(nodeBuilder.getName(), -1*changelayer);
					if(changeinp > 1)
						changeinputDimensionOfLayer(nodeBuilder.getName(), -1*changeinp);
				
			}
			
			
			
		}
		System.out.println("done");
		
	}
	
	public void deCompressInitializers()
	{
		Iterator<TensorProto> itInit= _graphBuilder.getInitializerList().iterator();
		int initIndex=0;
		while(itInit.hasNext())
		{
			TensorProto initializer = itInit.next();
			
			TensorProto.Builder decompressedBuilder = TensorProto.newBuilder();
			
			decompressedBuilder.setName(initializer.getName());
			
			for(int i=0; i<initializer.getDimsCount(); i++)
				decompressedBuilder.addDims((int)initializer.getDims(i));
			decompressedBuilder.setDataType(DataType.FLOAT);
			
			
			ByteString initializerData = getRawDataFromFloatData(initializer);
			
			byte[] ba = new byte[initializerData.size()];
			initializerData.copyTo(ba, 0, 0, initializerData.size());
			
			 Inflater inflater = new Inflater();  
			 inflater.setInput(ba);  
			try
			{
				 ByteArrayOutputStream outputStream = new ByteArrayOutputStream(ba.length);   
				 byte[] buffer = new byte[1024];  
				 
				 while (!inflater.finished()) 
				 {  
				    int count = inflater.inflate(buffer); // returns the generated code... index  
				    outputStream.write(buffer, 0, count);   
				 }  
			    outputStream.close(); 
			    byte[] output = outputStream.toByteArray(); 
			    ByteBuffer bb = ByteBuffer.allocate(output.length);
				
			    bb =  ByteBuffer.wrap(output);
			
			    bb.rewind();
			    decompressedBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
			}
			catch(Exception e)
			{
				System.err.println("ERROR COMPRESSING");
				return;
			}
			
			_graphBuilder.setInitializer(initIndex,decompressedBuilder.build() );
			initIndex++;
		}
	}

	public void compressInitializers()
	{
		Iterator<TensorProto> itInit= _graphBuilder.getInitializerList().iterator();
		int initIndex=0;
		while(itInit.hasNext())
		{
			TensorProto initializer = itInit.next();
			
			TensorProto.Builder compressedBuilder = TensorProto.newBuilder();
			
			compressedBuilder.setName(initializer.getName());
			
			for(int i=0; i<initializer.getDimsCount(); i++)
				compressedBuilder.addDims((int)initializer.getDims(i));
			compressedBuilder.setDataType(DataType.FLOAT);
			
			
			ByteString initializerData = getRawDataFromFloatData(initializer);
			
			byte[] ba = new byte[initializerData.size()];
			initializerData.copyTo(ba, 0, 0, initializerData.size());
			
			 Deflater deflater = new Deflater();  
			 deflater.setInput(ba);  
			try
			{
				 ByteArrayOutputStream outputStream = new ByteArrayOutputStream(ba.length);   
				 deflater.finish();  
				 byte[] buffer = new byte[1024];  
				 
				 while (!deflater.finished()) 
				 {  
				    int count = deflater.deflate(buffer); // returns the generated code... index  
				    outputStream.write(buffer, 0, count);   
				 }  
			    outputStream.close(); 
			    byte[] output = outputStream.toByteArray(); 
			    ByteBuffer bb = ByteBuffer.allocate(output.length);
				
			    bb =  ByteBuffer.wrap(output);
			
			    bb.rewind();
			    compressedBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
			}
			catch(Exception e)
			{
				System.err.println("ERROR COMPRESSING");
				return;
			}
			_graphBuilder.setInitializer(initIndex,compressedBuilder.build() );
			initIndex++;
		}
	}
	
	
	
	/* ******************************************************
	 * *******************PRIVATE FUNCTIONS******************
	 * ******************************************************
	 */
	
	

	private long getSumOfLongList( List<Long> numbers)
	{
		return numbers.stream().mapToLong(a->a.longValue()).sum();
	}
	
	private long getProductOfLongList( List<Long> numbers)
	{
		return (long)numbers.stream().mapToLong(a->a.longValue()).reduce((a,b)->a*b).getAsLong();
	}
	
	
	private Map<String, ONNX.TensorProto> getInitializerMap(GraphProto.Builder graphBuilder)
	{
		Map<String, ONNX.TensorProto> initializerMap = graphBuilder.getInitializerList().stream().
														collect(Collectors.toMap(tp-> tp.getName(),
																Function.identity()));
		
		//initializerMap.putAll(    graphBuilder.getInitializerList().stream().
		//												collect(Collectors.toMap(tp-> tp.getName().split("_0")[0],
	//															Function.identity()))   );
		
		//if(Configs.DATASET != "VOC")
		//	return initializerMap;
		List<NodeProto> allNodes = graphBuilder.getNodeList();
		Iterator<NodeProto> it = allNodes.iterator();
		while(it.hasNext())
		{
			NodeProto node = it.next();
			String opType = node.getOpType();
			switch(opType)
			{
			case "Conv":
				String weightName = node.getName()+"_weights";
				String biasName = node.getName()+"_bias";
				if(initializerMap.containsKey(node.getInput(1)) && (!initializerMap.containsKey(weightName)))
					initializerMap.put(weightName, initializerMap.get(node.getInput(1)));
				if(initializerMap.containsKey(node.getInput(2))  && (!initializerMap.containsKey(biasName)))
				    initializerMap.put(biasName, initializerMap.get(node.getInput(2)));
				break;
			case "BatchNormalization":
				String BNname = node.getName();
				String gammaName = ( BNname+ "_gamma");
				String betaName = (BNname + "_beta");
				String meanName = (BNname + "_mean");
				String varName = (BNname + "_var");
				
				if(initializerMap.containsKey(node.getInput(1)) && (!initializerMap.containsKey(gammaName)))
					initializerMap.put(gammaName, initializerMap.get(node.getInput(1)));
				if(initializerMap.containsKey(node.getInput(2)) && (!initializerMap.containsKey(betaName)))
				    initializerMap.put(betaName, initializerMap.get(node.getInput(2)));
				if(initializerMap.containsKey(node.getInput(3)) && (!initializerMap.containsKey(meanName)))
					initializerMap.put(meanName, initializerMap.get(node.getInput(3)));
				if(initializerMap.containsKey(node.getInput(4)) && (!initializerMap.containsKey(varName)))
					initializerMap.put(varName, initializerMap.get(node.getInput(4)));
				
				break;
			case "MatMul":
				String denseWeightName = node.getName()+"_weights";
				if(initializerMap.containsKey(node.getInput(1)) && (!initializerMap.containsKey(denseWeightName)))
					initializerMap.put(denseWeightName, transpose2DTensor(initializerMap.get(node.getInput(1))));
				break;
			case "Add":
				String checkBiasName = node.getInput(1);
				if(checkBiasName.contains("bias"))
				{
					String denseBiasName = checkBiasName.split("_")[0] + "_bias";
					if(initializerMap.containsKey(node.getInput(1)) && (!initializerMap.containsKey(denseBiasName)))
						initializerMap.put(denseBiasName, initializerMap.get(node.getInput(1)));
				}
				break;
			}
			
		}
		return initializerMap;
		
	}
	
	public  TensorProto transpose2DTensor(TensorProto tensorPro)
	{
		if(tensorPro.getDimsCount()!=2)
			return null;
		
		 int m = (int)tensorPro.getDims(1);
		 int n = (int)tensorPro.getDims(0);
		    
		TensorProto.Builder transposed =  TensorProto.newBuilder();
		transposed.addDims(m);
		transposed.addDims(n);
		
	
		transposed.setRawData(transpose2DrawData(tensorPro));

	    return transposed.build();
	}

	
	private ByteString transpose2DrawData(TensorProto tensorPro)
	{
		int row = (int)tensorPro.getDims(0);
		int col = (int)tensorPro.getDims(1);
		//byte[] matrix = new byte[row*col*4];
		ByteString initializerData = getRawDataFromFloatData(tensorPro);
		
		ByteBuffer rawbb = ByteBuffer.allocate(4*row*col);
		initializerData.copyTo(rawbb);
		rawbb.order(ByteOrder.LITTLE_ENDIAN);
		
		
		ByteBuffer transposedBuffer = ByteBuffer.allocate(row*col*4);
		transposedBuffer.order(ByteOrder.LITTLE_ENDIAN);
		
		int i = 0;
			for(int r=0; r<col; r++){
				for( int c=0; c<row; c++){ 

		    	transposedBuffer.putFloat(i*4, rawbb.getFloat(4*(r +  c*col))); //transpose
		    	i++;
		    }
		 }
		 
		   byte[] matrix = new byte[transposedBuffer.capacity()];
		   transposedBuffer.get(matrix);
	
		 //bb = ByteBuffer.wrap(matrix);
		 transposedBuffer.rewind();
		 
		 return ByteString.copyFrom(transposedBuffer, transposedBuffer.capacity());
	}
	
	
	private void updateInitializerWeightFirstDim(TensorProto.Builder tensorBuilder, int changeInNeurons)
	{
		long dims = tensorBuilder.getDims(0);
		//System.out.println("Dims: " + tensorBuilder.getDimsCount() + ":" + dims);
		

		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		int totalValues =1; //per neuron
		for(int i=1; i< tensorBuilder.getDimsCount(); i++)
		{
				long dim = tensorBuilder.getDims(i);
				totalValues*=dim;
		}
		
		if(changeInNeurons>0)
		{
			bb = ByteBuffer.allocate(initializerData.size() + changeInNeurons*totalValues*4);
			initializerData.copyTo(bb);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			bb.position(initializerData.size());
			for(int j=0;j<changeInNeurons;j++)
			{
				/*float a = new Float(Math.random()*0.01);
				if(Math.random() < 0.25)
					a*=-1;*/
				
				int randomNeuron = new Random().nextInt(new Long(tensorBuilder.getDims(0)).intValue());//TODO: replace with GAMAin's random
				
				byte[] minidim = new byte[totalValues*4];
				initializerData.copyTo(minidim, randomNeuron*totalValues*4, 0, totalValues*4);
				
				
				bb.put(minidim,0,minidim.length);//initializerData.size() + j*totalValues*4
				
			}
			
		}
		else
		{
			byte[] ba = new byte[initializerData.size()+ changeInNeurons*totalValues*4];
			initializerData.copyTo(ba,0,0,initializerData.size() + changeInNeurons*totalValues*4);
			bb =  ByteBuffer.wrap(ba);
		}
		
		bb.rewind();
		tensorBuilder.setDims(0, dims+changeInNeurons);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	}
	
	private List<Integer> pruneInitializerWeightFirstDim(TensorProto.Builder tensorBuilder, int changeInNeurons)
	{
		if(changeInNeurons>0)
		{
			return null; //Pruning not possible 
		}
		
		int mainDims = (int)tensorBuilder.getDims(0);
		//System.out.println("prune 1st Dims: " + tensorBuilder.getDimsCount() + ":" + tensorBuilder.getDims(0) + ":" + tensorBuilder.getDims(1));
		List<Integer> indexesToRemove;

		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		int totalValues =1; //per neuron
		for(int i=1; i< tensorBuilder.getDimsCount(); i++)
		{
				long dim = tensorBuilder.getDims(i);
				totalValues*=dim;
		}
		
		double[] sums = new double[mainDims];
		
		for(int j=0;j < mainDims;j++)
		{
			byte[] ba = new byte[totalValues*4];
			initializerData.copyTo(ba, j*4, 0, totalValues*4);
			ByteBuffer tempbb = ByteBuffer.allocate(totalValues*4);
			
			tempbb = ByteBuffer.wrap(ba);
			tempbb.order(ByteOrder.LITTLE_ENDIAN);
			
			float[] floatArray = new float[totalValues];
			FloatBuffer fb = tempbb.asFloatBuffer();
			
			fb.get(floatArray); //array();
			List<Double> dbstr =  IntStream.range(0, floatArray.length) .parallel().mapToDouble(i -> floatArray[i])
														.boxed().collect(Collectors.toList());
			sums[j] = dbstr.stream().reduce(0.0, Double::sum);
		}
			
	    indexesToRemove = IntStream.range(0, mainDims).boxed()
									.sorted(Comparator.comparing(i->sums[i])) //Sort indexes on primary array - which is sums
									.mapToInt(i->i).limit(Math.abs(changeInNeurons)).sorted() //Get lowest "valued" indexes
									.collect(ArrayList<Integer>::new, ArrayList::add, ArrayList::addAll);
		
		byte[] ba = new byte[initializerData.size()+ changeInNeurons*totalValues*4];
		int k=0;
		for(int j=0; j< mainDims; j++)
		{
			if(indexesToRemove.contains(new Integer(j)))
				continue; //DO-NOT-COPY
			
			initializerData.copyTo(ba,j*totalValues*4, k*totalValues*4, totalValues*4);
			k++;
		}
			
		bb =  ByteBuffer.wrap(ba);
		
		
		bb.rewind();
		tensorBuilder.setDims(0, mainDims+changeInNeurons);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		return indexesToRemove;
	}
	
	private void updateInitializerWeightSecondDim(TensorProto.Builder tensorBuilder, int changeInDim)
	{
		
		long numDims = tensorBuilder.getDimsCount();
		if(numDims <2)
			return;
		
		int firstDim = (int) tensorBuilder.getDims(0);
		int secondDim =(int) tensorBuilder.getDims(1);
		
		//System.out.println("Dims: " + tensorBuilder.getDimsCount() + ":" + firstDim);
		int displacementDim = 1;
		if(numDims > 2)
		{
			for(int i=2; i< numDims; i++)
			{
					long dim = tensorBuilder.getDims(i);
					displacementDim*=dim;
			}
		}

		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		byte[] ba = new byte[initializerData.size()+ (firstDim*changeInDim*displacementDim*4)];
		
		for(int i=0; i< firstDim;i++)
		{
			if(changeInDim>0)
			{
				ByteBuffer randomValueBuffer = ByteBuffer.allocate(changeInDim*displacementDim*4);
				
				randomValueBuffer.order(ByteOrder.LITTLE_ENDIAN);
				for(int j=0;j<changeInDim*displacementDim;j++)
				{
					float a = new Float(Math.random()*0.01);
					if(Math.random() < 0.25)
						a*=-1;
					randomValueBuffer.putFloat(j*4, a);
				}
				byte[] bufferToArray = new byte[randomValueBuffer.capacity()];
				randomValueBuffer.get(bufferToArray);
				
				initializerData.copyTo( ba,
										secondDim*displacementDim*i*4,//Source-offset
										(secondDim+changeInDim)*displacementDim*i*4,//target=offset
										secondDim*displacementDim*4);
				System.arraycopy(bufferToArray, 0, ba, 
									((secondDim+changeInDim)*displacementDim*i*4)+(secondDim*displacementDim*4),
									bufferToArray.length);
			}
			else
			{	
				initializerData.copyTo( ba,
										secondDim*displacementDim*i*4,//Source-offset
										(secondDim+changeInDim)*displacementDim*i*4,//target=offset
										(secondDim+changeInDim)*displacementDim*4);
				
			}
		}
			
		bb =  ByteBuffer.wrap(ba);
		bb.rewind();
		tensorBuilder.setDims(1, secondDim+changeInDim);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	}
	
	private void indexedPruneInitializerWeightSecondDim(TensorProto.Builder tensorBuilder, List<Integer> indexestoRemove, int DimChange)
	{
		
		int changeInDim = indexestoRemove.size();
		
		
		long numDims = tensorBuilder.getDimsCount();
		if(numDims <2)
			return;
		
		int firstDim = (int) tensorBuilder.getDims(0);
		int secondDim =(int) tensorBuilder.getDims(1);
		
		//System.out.println("2nd prune Dims: " + tensorBuilder.getDimsCount() + ":" + firstDim + ":" + secondDim + ":" + DimChange);
		int displacementValue = 1;//For FC layer it is 1
		if(Math.abs(DimChange)>changeInDim)
		{
			displacementValue = Math.floorDiv(Math.abs(DimChange),changeInDim);
			secondDim = Math.floorDiv(secondDim,displacementValue);
		}
		if(numDims > 2)
		{   for(int i=2; i< numDims; i++)
			{
					long dim = tensorBuilder.getDims(i);
					displacementValue*=dim;
			}
		}

		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		byte[] ba = new byte[initializerData.size()- (firstDim*changeInDim*displacementValue*4)];
		
		for(int i=0; i< firstDim;i++)
		{
			int k=0;
			for(int j=0;j<secondDim;j++)
			{
				if(indexestoRemove.contains(new Integer(j)))
					continue;
				initializerData.copyTo(ba, i*secondDim*displacementValue*4 + j*displacementValue*4, 
						                   i*(secondDim-changeInDim)*displacementValue*4 + k*displacementValue*4, displacementValue*4 );
				k++;
			}
		}
			
		bb =  ByteBuffer.wrap(ba);
		bb.rewind();
		tensorBuilder.setDims(1, (int) tensorBuilder.getDims(1) - Math.abs(DimChange));
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	}
	
	private void updateInitializerWeightKernelDim(TensorProto.Builder tensorBuilder, int kernelDim, int parentKernelDim)
	{
		long numDims = tensorBuilder.getDimsCount();
		if(numDims <4) 
			return;
		
		int firstDim = (int) tensorBuilder.getDims(0);
		int secondDim =(int) tensorBuilder.getDims(1);
		int squareKernelDim = kernelDim*kernelDim;
		int squareParentKernelDim = parentKernelDim*parentKernelDim;
		int squarechangeInDim = squareKernelDim - squareParentKernelDim;
	
		//System.out.println("Dims: " + tensorBuilder.getDimsCount() + ":" + firstDim);
		//int displacementDim = ;
		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		byte[] ba = new byte[ (firstDim*secondDim*squareKernelDim*4)];
		
		for(int i=0; i<(firstDim*secondDim);i++)
		{
			byte[] parentKernelArray = new byte[squareParentKernelDim*4];
			
			initializerData.copyTo(parentKernelArray,squareParentKernelDim*i*4,0,squareParentKernelDim*4);
			ByteBuffer parentKernelBuffer = ByteBuffer.wrap(parentKernelArray);
			parentKernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
			
			
			if(squarechangeInDim>0) //Expand
			{
				ByteBuffer kernelBuffer = ByteBuffer.allocate(kernelDim*kernelDim*4);
				kernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
				
				for(int j=0;j<kernelDim;j++)//1st kerneldim traversal
				{
					for(int k=0; k<kernelDim;k++)//2nd kerneldim traversal
					{
						if((j<parentKernelDim)&&(k<parentKernelDim)&&(j>0)&&(k>0)) //for 0th row or 0th column put random value
						{
							kernelBuffer.putFloat(parentKernelBuffer.getFloat((j*parentKernelDim + k)*4));
						}
						else
						{
							float a = (float)0.0; //new Float(Math.random()*0.01);
							//if(Math.random() < 0.25)
							//	a*=-1;
							kernelBuffer.putFloat(a);
						}
					}
				}
				
				System.arraycopy(kernelBuffer.array(), 0,
									ba,i*squareKernelDim*4,//destpos
									kernelBuffer.array().length);
			}
			else //shrink
			{	
				
				ByteBuffer kernelBuffer = ByteBuffer.allocate(kernelDim*kernelDim*4);
				kernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
				
				for(int j=0;j<kernelDim;j++)//1st kerneldim traversal
				{
					for(int k=0; k<kernelDim;k++)//2nd kerneldim traversal
					{
						kernelBuffer.putFloat(parentKernelBuffer.getFloat(((j+1)*parentKernelDim + (k+1))*4));
						
					}
				}
				
				System.arraycopy(kernelBuffer.array(), 0,
									ba,i*squareKernelDim*4,//destpos
									kernelBuffer.array().length);
			}
				
				
		}
			
		bb =  ByteBuffer.wrap(ba);
		bb.rewind();
		tensorBuilder.setDims(2, kernelDim);
		tensorBuilder.setDims(3, kernelDim);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
	}
	
	
	private void updateInitializerWeightKernelDim1D(TensorProto.Builder tensorBuilder, int kernelDim, int parentKernelDim)
	{
		long numDims = tensorBuilder.getDimsCount();
		if(numDims <4) 
			return;
		
		int firstDim = (int) tensorBuilder.getDims(0);
		int secondDim =(int) tensorBuilder.getDims(1);
		
		int changeInDim = kernelDim - parentKernelDim;
		
		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		byte[] ba = new byte[ (firstDim*secondDim*kernelDim*4)];
		
		for(int i=0; i<(firstDim*secondDim);i++)
		{
			byte[] parentKernelArray = new byte[parentKernelDim*4];
			
			initializerData.copyTo(parentKernelArray,parentKernelDim*i*4,0,parentKernelDim*4);
			ByteBuffer parentKernelBuffer = ByteBuffer.wrap(parentKernelArray);
			parentKernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
			
			
			if(changeInDim>0) //Expand
			{
				ByteBuffer kernelBuffer = ByteBuffer.allocate(kernelDim*4);
				kernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
				
				for(int j=0;j<kernelDim;j++)//1st kerneldim traversal
				{
				
					if((j<parentKernelDim)&&(j>0)) //when j=0 put zero value
					{
						kernelBuffer.putFloat( parentKernelBuffer.getFloat(j*4) );
					}
					else
					{
						float a = (float)0.00; 
						kernelBuffer.putFloat(a);
					}
			
				}
				
				System.arraycopy(kernelBuffer.array(), 0,
									ba,i*kernelDim*4,//destpos
									kernelBuffer.array().length);
			}
			else //shrink
			{	
				
				ByteBuffer kernelBuffer = ByteBuffer.allocate(kernelDim*4);
				kernelBuffer.order(ByteOrder.LITTLE_ENDIAN);
				
				for(int j=0;j<kernelDim;j++)//1st kerneldim traversal
				{
						kernelBuffer.putFloat(parentKernelBuffer.getFloat( (j+1)*4 ) );
				}
				
				System.arraycopy(kernelBuffer.array(), 0,
									ba,i*kernelDim*4,//destpos
									kernelBuffer.array().length);
			}
				
				
		}
			
		bb =  ByteBuffer.wrap(ba);
		bb.rewind();
		tensorBuilder.setDims(2, kernelDim);
		tensorBuilder.setDims(3, kernelDim);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
		
	}
	
	private void prune1DInitializer(TensorProto.Builder tensorBuilder, List<Integer> indexesToRemove )
	{
		long dims = tensorBuilder.getDims(0);
		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		int changeInNeurons = indexesToRemove.size();
		
		byte[] ba = new byte[initializerData.size() - changeInNeurons*4];
		int k=0;
		for(int j=0; j< dims; j++)
		{
			if(indexesToRemove.contains(new Integer(j)))
				continue; //DO-NOT-COPY
			
			initializerData.copyTo(ba,j*4, k*4,4);
			k++;
		}
		ByteBuffer bb;
		bb =  ByteBuffer.wrap(ba);
		
		bb.rewind();
		tensorBuilder.setDims(0, dims - changeInNeurons);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	}
	
	private void updateInitializerBias(TensorProto.Builder tensorBuilder, int changeInNeurons)
	{
		long dims = tensorBuilder.getDims(0);
		//System.out.println("Dims: " + tensorBuilder.getDimsCount() + ":" + dims);
		
		ByteString initializerData = getRawDataFromFloatData(tensorBuilder);
		ByteBuffer bb;
		
		if(changeInNeurons>0)
		{
			bb = ByteBuffer.allocate(initializerData.size() + changeInNeurons*4);
			initializerData.copyTo(bb);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			for(int j=0;j<changeInNeurons;j++)
			{
				float a = new Float(Math.random()*0.01);
				if(Math.random() < 0.25)
					a*=-1;
				bb.putFloat(initializerData.size() + j*4, a);
			}
			
		}
		else
		{
			byte[] ba = new byte[initializerData.size() + changeInNeurons*4];
			initializerData.copyTo(ba,0,0,initializerData.size() + changeInNeurons*4);
			bb =  ByteBuffer.wrap(ba);
		}
		
		bb.rewind();
		tensorBuilder.setDims(0, dims+changeInNeurons);
		tensorBuilder.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	}
	
	private void updateInputWeightFirstDim(String inputName, long totalNeurons)
	{
		List<ValueInfoProto.Builder> ibList = _graphBuilder.getInputBuilderList();
		Iterator<ValueInfoProto.Builder> it = ibList.iterator();
		
		while(it.hasNext())
		{
			ValueInfoProto.Builder inputBuilder = it.next();
			if(inputBuilder.getName().equals(inputName))
			{
				TensorShapeProto.Builder shapeBuilder = inputBuilder.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder();
				//Assuming there is always atleast 1 dimension! 
				shapeBuilder.getDimBuilder(0).setDimValue(totalNeurons);
				break;
			}
		}
	}
	
	private void updateInputWeightSecondDim(String inputName, long totalNeurons)
	{
		List<ValueInfoProto.Builder> ibList = _graphBuilder.getInputBuilderList();
		Iterator<ValueInfoProto.Builder> it = ibList.iterator();
		
		while(it.hasNext())
		{
			ValueInfoProto.Builder inputBuilder = it.next();
			if(inputBuilder.getName().equals(inputName))
			{
				TensorShapeProto.Builder shapeBuilder = inputBuilder.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder();
				if(shapeBuilder.getDimCount()>1)
					shapeBuilder.getDimBuilder(1).setDimValue(totalNeurons);
				break;
			}
		}
	}
	
	private void updateInputBias(String inputName, long totalNeurons)
	{
		List<ValueInfoProto.Builder> ibList = _graphBuilder.getInputBuilderList();
		Iterator<ValueInfoProto.Builder> it = ibList.iterator();
		
		while(it.hasNext())
		{
			ValueInfoProto.Builder inputBuilder = it.next();
			if(inputBuilder.getName().equals(inputName))
			{
				TensorShapeProto.Builder shapeBuilder = inputBuilder.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder();
				shapeBuilder.getDimBuilder(0).setDimValue(totalNeurons);
				break;
			}
		}
	}
	
	
	private ByteString getRawDataFromFloatData(TensorProto tensorInit)
	{
		if(tensorInit.hasRawData())
			return tensorInit.getRawData();
		else
		{
			List<Float> values  = tensorInit.getFloatDataList();
			
			float[] floatArr = new float[values.size()];
			int index = 0;
			for (final Float value: values) {
				floatArr[index++] = value;
			}
			
			ByteBuffer buffer = ByteBuffer.allocate(4 * values.size());
			buffer.order(ByteOrder.LITTLE_ENDIAN);
			    for (float value : floatArr){
			        buffer.putFloat(value);
			    }
			buffer.rewind();
			ByteString bs = ByteString.copyFrom(buffer);
			return bs;
		}
	}
	
	

	private ByteString getRawDataFromFloatData(TensorProto.Builder tensorInit)
	{
		if(tensorInit.hasRawData())
			return tensorInit.getRawData();
		else
		{
			List<Float> values  = tensorInit.getFloatDataList();
			
			float[] floatArr = new float[values.size()];
			int index = 0;
			for (final Float value: values) {
				floatArr[index++] = value;
			}
			
			ByteBuffer buffer = ByteBuffer.allocate(4 * floatArr.length);
			buffer.order(ByteOrder.LITTLE_ENDIAN);

			    for (float value : floatArr){
			        buffer.putFloat(value);
			    }
			buffer.rewind();
			ByteString bs = ByteString.copyFrom(buffer);
			return bs;
		}
	}
	
	

}
