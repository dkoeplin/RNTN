import optiml.compiler._
import optiml.library._
import optiml.shared._

trait RNTNOps extends OptiMLApplication with Utilities {
	
	private val TREECOLS = 5
	private val PARENT = 0  		// Gives the index of the parent node of the node
	private val LKID   = 1  		// Gives the index of the left child of the node
	private val RKID   = 2  		// Gives the index of the right child of the node
	private val WORD   = 3       	// Gives the phrase index represented by the node
	private val LABEL  = 4       	// Gives the predetermined sentiment score for the node

	private val POSNEGNEUT = 5

	private val WORDI 		  = true
	//private val QRNN  		  = true
	private val NUMCLASSES 	  = 5
	private val NEUTRALLABEL  = if (NUMCLASSES == 2)  POSNEGNEUT else (NUMCLASSES - 1)/2
	//private val NEUTRALWEIGHT = if (NUMCLASSES == 2)  0.0		 else 1.0  	// 0.3 as default?
	private val WORDSIZE      = 25

	private val ADAZERO = true
	//private val ADAGRAD = true
	private val LR		= 0.01
	private val ADAEPS  = 0.001

	private val regC_Wc = 0.0001
	private val regC_W	= 0.1
	private val regC_Wt = 0.15
	private val regC_Wv	= 0.0001

	private val VERBOSE = false

	// -------------------------------------------------------------------------------------//
	//										Activation of Node						  		//
	// -------------------------------------------------------------------------------------//
	def computeNode(
		node: 	  Rep[DenseVector[Int]],
		kidsActs: Rep[DenseVector[DenseVector[Double]]],
		Wc:   	  Rep[DenseMatrix[Double]],
		W:    	  Rep[DenseMatrix[Double]],
		Wt:       Rep[DenseMatrix[Double]],
		Wv:   	  Rep[DenseMatrix[Double]]
	) = {
		val lKid = node(LKID)
		val act = 	if 		(lKid == -1 && WORDI) { Wv(node(WORD)).t }
					else if (lKid == -1)	  	  { Wv(node(WORD)).map(d => tanh(d)).t }
					else {
						val ab = kidsActs(lKid) << kidsActs(node(RKID))
						val quad = 	(0::WORDSIZE) {
						  				row => ((ab.t * get3D(Wt, 2*WORDSIZE, row)) * ab.t).sum 
						  			}
						(W * (ab << 1.0) + quad).map(d => tanh(d))
					}	
		val out = softmaxVect(Wc * (act << 1.0))
		(act, out)
	}

	// -------------------------------------------------------------------------------------//
	//						Tensor Cost Function (Runs Batch of Trees)				  		//
	// -------------------------------------------------------------------------------------//
	def trainBatch (
		trees: Rep[DenseVector[DenseVector[DenseMatrix[Int]]]],
		Wc:    Rep[DenseMatrix[Double]],				// NUMCLASSES X (WORDSIZE + 1)
		W:	   Rep[DenseMatrix[Double]],				// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:    Rep[DenseMatrix[Double]],				// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:	   Rep[DenseMatrix[Double]] 				// WORDSIZE X allPhrases.length
	) = {
		val curBatchSize = trees.length
		val dWc_batch = DenseVector[DenseMatrix[Double]](curBatchSize, true)
		val dW_batch  = DenseVector[DenseMatrix[Double]](curBatchSize, true)
		val dWt_batch = DenseVector[DenseMatrix[Double]](curBatchSize, true)
		val dWv_batch = DenseVector[DenseMatrix[Double]](curBatchSize, true)

		tic("Batch Run")
		val costs_batch = (0::curBatchSize) { t => 
			verbosePrint("		Training on tree " + t, VERBOSE)
			tic("Train tree")
			val tree = trees(t)
			val levels   = tree.length
			val maxLevel = levels - 1
			val largestLevel = tree.map(level => level.numRows).max // Find largest level in this tree

			val acts = DenseVector[DenseVector[DenseVector[Double]]](levels, true)
			val outs = DenseVector[DenseVector[DenseVector[Double]]](levels, true)
			//val cost = DenseVector[Double](levels, true)
			val dWc = DenseMatrix[Double](NUMCLASSES, WORDSIZE + 1)
			val dW  = DenseMatrix[Double](WORDSIZE, WORDSIZE*2 + 1)
			val dWt = DenseMatrix[Double](WORDSIZE*WORDSIZE*2, WORDSIZE*2)
			val dWv = DenseMatrix[Double](Wv.numRows, WORDSIZE)

			var select_deltas_1 = true			
			val deltas_1 = DenseVector[DenseVector[Double]](largestLevel, true)
			val deltas_2 = DenseVector[DenseVector[Double]](largestLevel, true)
			deltas_1(0) = DenseVector.zeros(WORDSIZE).t
			var n = 0
			// -------------------------------------------------------------------------------------//
			//								Forward Propagation of Tree						  		//
			// -------------------------------------------------------------------------------------//
			verbosePrint("			Activating tree with " + levels + " level(s)", VERBOSE)	

			var curLevel = maxLevel
			while (curLevel >= 0) {
				verbosePrint("				Activating level " + curLevel, VERBOSE)	
				val kidsActs = if (curLevel == maxLevel) acts(curLevel) else acts(curLevel + 1)
				val level 	 = tree(curLevel)
				val numNodes = level.numRows
				
				val levelIO = (0::numNodes) { n =>
					verbosePrint("					Activating node " + n, VERBOSE)	
					val node = level(n)
					pack(computeNode(node, kidsActs, Wc, W, Wt, Wv))
					//val nodeLabel = node(LABEL)
					//val cost = log(out(nodeLabel))
					//pack((act, out, cost))
				}
				acts(curLevel) = levelIO.map(_._1)
				outs(curLevel) = levelIO.map(_._2)
				//cost(curLevel) = levelIO.map(_._3).sum
				curLevel -= 1
			}
			//val treeCost = -cost.sum

			// -------------------------------------------------------------------------------------//
			//								Backward Propagation of Tree					  		//
			// -------------------------------------------------------------------------------------//
			verbosePrint("			Running backward propagation on tree", VERBOSE)

			curLevel = 0
			while (curLevel < levels) {
				verbosePrint("				Backpropagating at level " + curLevel, VERBOSE)	
				val level  = tree(curLevel)
				val levelActs = acts(curLevel)
				val levelOuts = outs(curLevel)
				val levelAbove = min(curLevel + 1, maxLevel)	// level of children
				
				val kidsActs  = acts(levelAbove)
				val kidsLkids = tree(levelAbove).getCol(LKID)

				val numNodes   = level.numRows
				val numParents = if (curLevel == maxLevel) { 0 }
							     else if (curLevel == 0)   { 1 }
								 else {level.getCol(WORD).count(d => d == -1)}
				val numLeaves  = numNodes - numParents	

				verbosePrint("				Back-propagating the " + numNodes + " nodes at level " + curLevel, VERBOSE)
				n = 0
				while (n < numNodes) {
					verbosePrint("					Backpropagating at node " + n, VERBOSE)	
					val node  	   = level(n)
					val nodeAct    = levelActs(n)		// WORDSIZE X 1
					val nodeOut    = levelOuts(n)	    // NUMCLASSES X 1
					val deltaFromParent = if (select_deltas_1) deltas_1(n) else deltas_2(n)
					
					val lKid = node(LKID)
					val nodeLabel = node(LABEL) 

					verbosePrint("					Calculating deltaDown and dWc", VERBOSE)
					val errorCats = (0::NUMCLASSES) { num => if (num == nodeLabel) { nodeOut(num) - 1 } else { nodeOut(num) } }
					val deltaDownCatOnly = Wc.t * errorCats.t
					val deltaDownAddCat  = if (lKid == -1 && WORDI) { deltaDownCatOnly * (nodeAct << 1.0) }
										   else 		   			{ deltaDownCatOnly * (nodeAct << 1.0).map(d => 1 - d*d) }

		            val deltaDownFull = deltaFromParent.t + deltaDownAddCat(0::WORDSIZE)
		           	dWc += errorCats.t.toMat * (nodeAct.t << 1.0).toMat

					if (lKid != -1) {
						val rKid = node(RKID)

						val ab  = kidsActs(lKid) << kidsActs(rKid)	// 2 * WORDSIZE X 1 (column vector)
						verbosePrint("					Calculating dWt", VERBOSE)
						val dWtbase = ab.toMat * ab.t.toMat   
						dWt += pack3D( (0::WORDSIZE) {row => deltaDownFull(row) * dWtbase } )

						verbosePrint("					Calculating dW", VERBOSE)
						dW += deltaDownFull.t.toMat * (ab.t << 1.0).toMat  // WORDSIZE X 1 * 1 X WORDSIZE 

						verbosePrint("					Calculating delta to children", VERBOSE)
						val linearPart = W.t * deltaDownFull.t
						val quadParts = (0::WORDSIZE) {row => 
											val Wt_row = get3D(Wt, 2*WORDSIZE, row)
											(Wt_row + Wt_row.t) * (deltaDownFull(row) * ab) 
										}
						val deltaDownBothVec = linearPart(0::2*WORDSIZE) + quadParts.sum

						// Pass on errors to each child
						if (select_deltas_1) {
							deltas_2(lKid) = if (kidsLkids(lKid) == -1 && WORDI) { deltaDownBothVec(0::WORDSIZE) }
											 else { deltaDownBothVec(0::WORDSIZE) * ab(0::WORDSIZE).map(d => 1 - d*d) }
						
							deltas_2(rKid) = if (kidsLkids(rKid) == -1 && WORDI) { deltaDownBothVec(WORDSIZE::2*WORDSIZE) }
											 else { deltaDownBothVec(WORDSIZE::2*WORDSIZE) * ab(WORDSIZE::2*WORDSIZE).map(d => 1 - d*d) }
						}
						else {
							deltas_1(lKid) = if (kidsLkids(lKid) == -1 && WORDI) { deltaDownBothVec(0::WORDSIZE) }
											 else { deltaDownBothVec(0::WORDSIZE) * ab(0::WORDSIZE).map(d => 1 - d*d) }
						
							deltas_1(rKid) = if (kidsLkids(rKid) == -1 && WORDI) { deltaDownBothVec(WORDSIZE::2*WORDSIZE) }
											 else { deltaDownBothVec(WORDSIZE::2*WORDSIZE) * ab(WORDSIZE::2*WORDSIZE).map(d => 1 - d*d) }
						}
					}
					else {
						dWv(node(WORD)) = dWv(node(WORD)) + deltaDownFull
					}
					n += 1
				}
				select_deltas_1 = !select_deltas_1
				curLevel += 1
			}	
			verbosePrint("			Completed backward propagation of tree", VERBOSE)

			toc("Train tree")
			dWc_batch(t) = dWc
			dW_batch(t)  = dW
			dWt_batch(t) = dWt
			dWv_batch(t) = dWv
			//(treeCost)
		}
		verbosePrint("		Trees completed", VERBOSE)
		toc("Batch Run")

		verbosePrint("		Computing final deltas", VERBOSE)
		tic("Final computation")

		val numSent = curBatchSize.toDouble
		val dfWc = dWc_batch.sum * (1/numSent) + (Wc * regC_Wc)
		val dfW  = dW_batch.sum * (1/numSent) + ( (W(*, 0::2*WORDSIZE) * regC_W ) <<| DenseVector.zeros(WORDSIZE) ) 
 		val dfWt = dWt_batch.sum * (1/numSent) + Wt * regC_Wt
		val dfWv = dWv_batch.sum * (1/numSent) + Wv * regC_Wv
		toc("Final computation")

		(dfWc, dfW, dfWt, dfWv)
	}	// end of def tensorCostFunction

	// -------------------------------------------------------------------------------------//
	//									Create Batch of Trees					 	 		//
	// -------------------------------------------------------------------------------------//
	def createBatch(
        trees:     Rep[DenseVector[DenseVector[DenseMatrix[Int]]]],
        Wv:        Rep[DenseMatrix[Double]],
        batchIter: Rep[Int],
        batchSize: Rep[Int],
        indices:   Rep[IndexVector]
    ) = {
    	val beginBatch   = batchIter*batchSize
       	val endBatch     = min(beginBatch + batchSize, trees.length)
      	val batchIndices = indices.slice(beginBatch, endBatch)

       	val unmappedTrees = trees(batchIndices)

       	verbosePrint("Finding all words in this batch", VERBOSE)
      	val batchInds = IndexVector(unmappedTrees.flatMap(tree => tree.flatMap(level => level.getCol(WORD).filter(d => d >= 0))).distinct)

      	verbosePrint("Creating batch Wv", VERBOSE)
     	val batchWv = Wv(batchInds)

      	// map phrase numbers to batchWv based on filtered word indices
       	verbosePrint("Creating batch map", VERBOSE)
       	val batchMap = DenseVector[Int](batchInds.max + 1, true)
      	batchMap.update(batchInds, 1::batchInds.length+1)
      	batchMap -= 1

      	verbosePrint("Mapping trees", VERBOSE)
     	val batchTrees = unmappedTrees.map( tree => 
        	tree.map( level =>
            	(0::level.numRows, 0::TREECOLS) { (node, col) =>
                	if (col == WORD) { 
                		val word = level(node, WORD)
                		if (word >= 0) { batchMap(word) } else { word } 
                	}
                    else             { level(node, col) }
            	}
            ) 
        )
        /* (0::unmappedTrees.length) { t =>
        	val tree = unmappedTrees(t)
			(0::t.length) { l =>
				val level = t(l)
				(*, 0::TREECOLS) {
					if (col == WORD) {level.getCol(WORD).map(w => if (w >= 0) {batchMap(w)} else w)}
					else 			 {level.getCol(col)}
				}	
			}
        } */
        (batchTrees, batchWv, batchInds, batchMap)
    }

	// -------------------------------------------------------------------------------------//
	//								Batch Training of Trees						 	 		//
	// -------------------------------------------------------------------------------------//
	def trainOnTrees(
		trees: 		Rep[DenseVector[DenseVector[DenseMatrix[Int]]]],
		Wc:   		Rep[DenseMatrix[Double]],	// NUMCLASSES X (WORDSIZE + 1)
		W:	  		Rep[DenseMatrix[Double]],	// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:   		Rep[DenseMatrix[Double]],	// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:		   	Rep[DenseMatrix[Double]], 	// WORDSIZE X allPhrases.length
		ssWc:   	Rep[DenseMatrix[Double]],	// NUMCLASSES X (WORDSIZE + 1)
		ssW:	  	Rep[DenseMatrix[Double]],	// WORDSIZE X (WORDSIZE * 2 + 1)
		ssWt:   	Rep[DenseMatrix[Double]],	// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		ssWv:		Rep[DenseMatrix[Double]], 	// WORDSIZE X allPhrases.length
		batchSize: 	Rep[Int],
		numBatches: Rep[Int]
	) {
		val numTrees   = trees.length
		println("Running " + numTrees + " trees in " + numBatches + " batches")

		// Randomize train set order - necessary for good performance w/ SGD
		val trainIndices = randperm(numTrees, numTrees)

		var batchIter = 0
		while (batchIter < numBatches) {
			//println("	Starting batch " + (batchIter + 1) + "/" + numBatches)
			// Randomized batches from the dataset
			//val (batchTrees, batchWv, batchInds, batchMap) = createBatch(trees, Wv, batchIter, batchSize, randomTrain)
			val beginBatch   = batchIter*batchSize
			val endBatch     = min(beginBatch + batchSize, trees.length)
			val batchIndices = trainIndices.slice(beginBatch, endBatch)
			val batchTrees = trees(batchIndices)

			verbosePrint("		Running trees...", VERBOSE)
			val (dfWc, dfW, dfWt, dfWv) = trainBatch(batchTrees, Wc, W, Wt, Wv)

			verbosePrint("		[TRAINING] Updating sums of squares...", VERBOSE)
			if (ADAZERO && batchIter == 0) {
				setMatrix(ssWc, dfWc.map(e => e*e))
				setMatrix(ssW, dfW.map(e => e*e))
				setMatrix(ssWt, dfWt.map(e => e*e))
				setMatrix(ssWv, dfWv.map(e => e*e))
			}
			else {
				ssWc += dfWc.map(e => e*e)
				ssW  += dfW.map(e => e*e)
				ssWt += dfWt.map(e => e*e)
				ssWv += dfWv.map(e => e*e)
			}

			verbosePrint("		[TRAINING] Calculating new weights...", VERBOSE)
			Wc -= ((dfWc * LR) / ssWc.map(e => sqrt(e) + ADAEPS))
			W  -= ((dfW  * LR) / ssW.map(e => sqrt(e) + ADAEPS))
			Wt -= ((dfWt * LR) / ssWt.map(e => sqrt(e) + ADAEPS))
			Wv -= ((dfWv * LR) / ssWv.map(e => sqrt(e) + ADAEPS))

			batchIter += 1
		}	// end of while loop
	}

	// -------------------------------------------------------------------------------------//
	//							Create Batches of Trees for Evaluation			 	 		//
	// -------------------------------------------------------------------------------------//
	def createEvalBatches(
        trees:      Rep[DenseVector[DenseVector[DenseMatrix[Int]]]],
        batchSize:  Rep[Int]
    ) = {
    	val numTrees = trees.length
    	val numBatches = ceil(numTrees.toDouble/batchSize)
		val batches = (0::numBatches) { i =>
      		val batchIndices = (i*batchSize)::min((i+1)*batchSize, numTrees)
       		val unmappedTrees = trees(batchIndices)

      		val batchInds = IndexVector(unmappedTrees.flatMap(tree => tree.flatMap(level => level.getCol(WORD).filter(d => d >= 0))).distinct)

      		// map phrase numbers to future batched Wv based on filtered word indices
       		verbosePrint("Creating batch map", VERBOSE)
       		val batchMap = DenseVector[Int](batchInds.max + 1, true)
      		batchMap.update(batchInds, 1::batchInds.length+1)
      		batchMap -= 1

      		verbosePrint("Mapping trees", VERBOSE)
     		val batchTrees = unmappedTrees.map( tree => 
        		tree.map( level =>
            		(0::level.numRows, 0::TREECOLS) { (node, col) =>
                		if (col == WORD) { 
                			val word = level(node, WORD)
                			if (word >= 0) { batchMap(word) } else { word } 
                		}
                    	else             { level(node, col) }
            		}
            	)	 
        	)
        	/* (0::unmappedTrees.length) { t =>
        	val tree = unmappedTrees(t)
				(0::t.length) { l =>
					val level = t(l)
					(*, 0::TREECOLS) {
						if (col == WORD) {level.getCol(WORD).map(w => if (w >= 0) {batchMap(w)} else w)}
						else 			 {level.getCol(col)}
					}	
				}
	        } */
  			pack(batchTrees, batchInds)
        }
        (batches.map(_._1), batches.map(_._2))
    }

	// -------------------------------------------------------------------------------------//
	//									Activation of Tree							  		//
	// -------------------------------------------------------------------------------------//
	def activateTree(
		tree: Rep[DenseVector[DenseMatrix[Int]]],
		Wc:   Rep[DenseMatrix[Double]],			// NUMCLASSES X (WORDSIZE + 1)
		W:	  Rep[DenseMatrix[Double]],			// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:   Rep[DenseMatrix[Double]],			// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:	  Rep[DenseMatrix[Double]] 			// WORDSIZE X allPhrases.length
	) = {
		val levels   = tree.length
		val maxLevel = levels - 1
		
		val outs = DenseVector[DenseVector[DenseVector[Double]]](levels, true)

		var curLevel = maxLevel
		var kidsActs = DenseVector[DenseVector[Double]]()
		while (curLevel >= 0) {
			val level 	 = tree(curLevel)
			val numNodes = level.numRows
			
			val levelIO = (0::numNodes) { n =>
				pack(computeNode(level(n), kidsActs, Wc, W, Wt, Wv))
			} 
			outs(curLevel) = levelIO.map(_._2)
			
			curLevel -= 1
			kidsActs = levelIO.map(_._1)	// return activation for next stage of loop (next level of tree)
		}
		(outs)	// return outputs
	}

	// -------------------------------------------------------------------------------------//
	//						Activate all Trees for Accuracy Checking					  	//
	// -------------------------------------------------------------------------------------//
	def evalOnTrees(
		batches: Rep[DenseVector[DenseVector[DenseVector[DenseMatrix[Int]]]]],
		inds: 	 Rep[DenseVector[IndexVector]],
		Wc:      Rep[DenseMatrix[Double]],				// NUMCLASSES X (WORDSIZE + 1)
		W:	     Rep[DenseMatrix[Double]],				// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:      Rep[DenseMatrix[Double]],				// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:	     Rep[DenseMatrix[Double]] 				// WORDSIZE X allPhrases.length
	) {
		tic("All Batches")
		val results = (0::batches.length) { i =>
			verbosePrint("	Starting batch " + (i + 1) + "/" + numBatches, VERBOSE)
			tic("Current Batch")

			val batchTrees = batches(i)
			val allNodesLabels = batchTrees.map(tree => tree.flatMap(level => level.getCol(LABEL)))
			val nodeBinaryTotal = allNodesLabels.map(tree => tree.count(label => label != NEUTRALLABEL)).sum
			val rootBinaryTotal = allNodesLabels.count(tree => tree(0) != NEUTRALLABEL)

			val curBatchSize = batchTrees.length
			verbosePrint("		Running trees...", VERBOSE)
			val treesCorrect = (0::curBatchSize) { t =>
				verbosePrint("		Activating tree " + t, VERBOSE)
				tic("Tree Activation")
				val outs = DenseVector.flatten( activateTree(batchTrees(t), Wc, W, Wt, Wv(inds(i)) ) )
				toc("Tree Activation")
				val binaryActs = outs.map(output => output(3) + output(4) > output(0) + output(1))	// assumes NUMCLASSES = 5
				val labelsBinaryMatch = binaryActs.zip(allNodesLabels(t)) {(l1, l2) => (l1 && (l2 > NEUTRALLABEL)) || (!l1 && (l2 < NEUTRALLABEL))}
				
				val nodesBinaryCorrect = labelsBinaryMatch.count(d => d)
				val rootBinaryCorrect  = if (labelsBinaryMatch(0)) {1} else {0}
				pack( (nodesBinaryCorrect, rootBinaryCorrect) )
			}

			val correctNodes = treesCorrect.map(_._1).sum
			val correctRoots = treesCorrect.map(_._2).sum

			if (VERBOSE) {
				val percentRootsBatch = correctRoots.toDouble / rootBinaryTotal
				val percentNodesBatch = correctNodes.toDouble / nodeBinaryTotal
				println("----------------- Batch complete -----------------")
				println("Root binary accuracy over all trees in batch: " + percentRootsBatch + "  ( " + correctRoots + "/" + rootBinaryTotal + " )")	
				println("Node binary accuracy over all trees in batch: " + percentNodesBatch + "  (" + correctNodes + "/" + nodeBinaryTotal + " )")
				println("--------------------------------------------------")
			}
			
			toc("Current Batch")
			pack( (correctNodes, correctRoots, nodeBinaryTotal, rootBinaryTotal) )
		}
		val correctNodes = results.map(_._1).sum
		val correctRoots = results.map(_._2).sum
		val numNodes     = results.map(_._3).sum
		val numRoots     = results.map(_._4).sum
		val percentRoots = correctRoots.toDouble / numRoots
		val percentNodes = correctNodes.toDouble / numNodes

		toc("All Batches")
		println("Root binary accuracy over all trees:	" + percentRoots + "	( " + correctRoots + "/" + numRoots + " )")
		println("Node binary accuracy over all trees:	" + percentNodes + "	( " + correctNodes + "/" + numNodes + " )")
		(percentRoots, percentNodes)
	}

}