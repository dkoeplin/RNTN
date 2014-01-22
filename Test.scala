import optiml.compiler._
import optiml.library._
import optiml.shared._

object TestInterpreter extends OptiMLApplicationInterpreter with Test
object TestCompiler extends OptiMLApplicationCompiler with Test
trait Test extends OptiMLApplication with Utilities{
	
	private val TREECOLS = 6
	private val PARENT = 0  		// Gives the index of the parent node of the node
	private val LKID   = 1  		// Gives the index of the left child of the node
	private val RKID   = 2  		// Gives the index of the right child of the node
	private val WORD   = 3       	// Gives the phrase index represented by the node
	private val LABEL  = 4       	// Gives the predetermined sentiment score for the node
	private val LEAFS  = 5       	// Gives the number of leaves the node is an ancestor of

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

	private val regC_Wc 	  = 0.0001
	private val regC_W	   	  = 0.1
	private val regC_Wt 	  = 0.15
	private val regC_Wv		  = 0.0001

	private val VERBOSE = false

	/*if (VERBOSE) {
		println("dWc sum...")
		val dWc = dWc_level.sum
		println(dWc.numRows + " X " + dWc.numCols)
		println(dWc_tree.numRows + " X " + dWc_tree.numCols)
		if (numParents > 0) {
			println("dW sum...")
			val dW = dW_level.sum
			println(dW.numRows + " X " + dW.numCols)
			println(dW_tree.numRows + " X " + dW_tree.numCols)
			println("dWt sum...")
			val dWt = dWt_level.sum
			println(dWt.numRows + " X " + dWt.numCols)
			println(dWt_tree.numRows + " X " + dWt_tree.numCols)
		}
		if (numLeaves > 0) {
			println("dWv sum...")
			val dWv = dWv_level.sum
			println(dWv.numRows + " X " + dWv.numCols)
			println(dWv_tree.numRows + " X " + dWv_tree.numCols)
		}
	}*/

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
		val act = 	if 		(lKid == -1 && WORDI) { Wv.getCol(node(WORD)).toDense }
					else if (lKid == -1)	  	   { Wv.getCol(node(WORD)).map(d => tanh(d)) }
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
	//								Forward Propagation of Tree						  		//
	// -------------------------------------------------------------------------------------//
	def forwardPropTree(
		tree: Rep[DenseVector[DenseMatrix[Int]]],
		Wc:   Rep[DenseMatrix[Double]],			// NUMCLASSES X (WORDSIZE + 1)
		W:	  Rep[DenseMatrix[Double]],			// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:   Rep[DenseMatrix[Double]],			// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:	  Rep[DenseMatrix[Double]] 			// WORDSIZE X allPhrases.length
	) = {
		val levels   = tree.length
		val maxLevel = levels - 1
		verbosePrint("			Activating tree with " + levels + " level(s)", VERBOSE)	

		val acts = DenseVector[DenseVector[DenseVector[Double]]](levels, true)
		val outs = DenseVector[DenseVector[DenseVector[Double]]](levels, true)
		val cost = DenseVector[Double](levels, true)

		var curLevel = maxLevel
		while (curLevel >= 0) {
			verbosePrint("				Activating level " + curLevel, VERBOSE)	
			val kidsActs = if (curLevel == maxLevel) acts(curLevel) else acts(curLevel + 1)
			val level 	 = tree(curLevel)
			val numNodes = level.numRows
			
			val levelIO = (0::numNodes) { n =>
				verbosePrint("					Activating node " + n, VERBOSE)	
				val node = level(n)
				val (act, out) = computeNode(node, kidsActs, Wc, W, Wt, Wv)
				val nodeLabel = node(LABEL)
				val cost = log(out(nodeLabel))
				pack((act, out, cost))
			}
			acts(curLevel) = levelIO.map(_._1)
			outs(curLevel) = levelIO.map(_._2)
			cost(curLevel) = levelIO.map(_._3).sum
			curLevel -= 1
		}
		(acts, outs, -cost.sum)
	}

	def backwardPropTree (
		tree: Rep[DenseVector[DenseMatrix[Int]]],
		Wc:   Rep[DenseMatrix[Double]],					// NUMCLASSES X (WORDSIZE + 1)
		W:	  Rep[DenseMatrix[Double]],					// WORDSIZE X (WORDSIZE * 2 + 1)
		Wt:   Rep[DenseMatrix[Double]],				  	// WORDSIZE X [(WORDSIZE * 2) X (WORDSIZE * 2)]
		Wv:	  Rep[DenseMatrix[Double]], 				// WORDSIZE X allPhrases.length
		acts: Rep[DenseVector[DenseVector[DenseVector[Double]]]],
		outs: Rep[DenseVector[DenseVector[DenseVector[Double]]]]
	) = {
		verbosePrint("			Running backward propagation on tree", VERBOSE)

		val levels   = tree.length
		val maxLevel = levels - 1

		val dWc_tree = DenseMatrix[Double](NUMCLASSES, WORDSIZE + 1)
		val dW_tree  = DenseMatrix[Double](WORDSIZE, WORDSIZE*2 + 1)
		val dWt_tree = DenseMatrix[Double](WORDSIZE*WORDSIZE*2, WORDSIZE*2)
		val dWv_tree = DenseMatrix[Double](WORDSIZE, Wv.numCols)

		implicit def diff(t1: Rep[DenseVector[DenseVector[Double]]], t2: Rep[DenseVector[DenseVector[Double]]]): Rep[Double] = 1.0

		var curLevel = 0
		untilconverged(DenseVector[DenseVector[Double]](DenseVector.zeros(WORDSIZE).t), tol=0.0, maxIter=levels) { deltas =>
			verbosePrint("				Backpropagating at level " + curLevel, VERBOSE)	
			val level  = tree(curLevel)
			val levelActs = acts(curLevel)
			val levelOuts = outs(curLevel)
			val levelAbove = min(curLevel + 1, maxLevel)	// level of children
			
			val kidsActs	  = acts(levelAbove)
			val kidsLeafs	  = tree(levelAbove).getCol(LEAFS)

			val numNodes   = level.numRows
			val numParents = if (curLevel == maxLevel) { 0 }
						     else if (curLevel == 0)   { 1 }
							 else {level.getCol(LEAFS).count(d => d > 1)}
			val numLeaves  = numNodes - numParents

			val deltas_level = DenseVector[DenseVector[Double]](numParents*2, true)
			//val dWc_level = DenseVector[DenseMatrix[Double]](numNodes, true)
			val dW_level  = DenseVector[DenseMatrix[Double]](numParents, true)
			val dWt_level = DenseVector[DenseMatrix[Double]](numParents, true)
			val dWv_level = DenseVector[DenseMatrix[Double]](numLeaves, true)

			verbosePrint("				Back-propagating the " + numNodes + " nodes at level " + curLevel, VERBOSE)
			val dWc_level = (0::numNodes) {n =>
				verbosePrint("					Backpropagating at node " + n, VERBOSE)	
				val node  	   = level(n)
				val nodeAct    = levelActs(n)		// WORDSIZE X 1
				val nodeOut    = levelOuts(n)	    // NUMCLASSES X 1
				val deltaFromParent = deltas(n)
				
				val lKid = node(LKID)
				val nodeLabel = node(LABEL) 
				// Removed neutral label weight and "node usefulness" stuff from here

				verbosePrint("					Calculating deltaDown and dWc", VERBOSE)
				val errorCats = (0::NUMCLASSES) { num => if (num == nodeLabel) { nodeOut(num) - 1 } else { nodeOut(num) } }
				val deltaDownCatOnly = Wc.t * errorCats.t
				val deltaDownAddCat  = if (lKid == -1 && WORDI) { deltaDownCatOnly * (nodeAct << 1.0) }
									   else 		   			{ deltaDownCatOnly * (nodeAct << 1.0).map(d => 1 - d*d) }

	            val deltaDownFull = deltaFromParent.t + deltaDownAddCat(0::WORDSIZE)
	            val dWc = errorCats.t.toMat * (nodeAct.t << 1.0).toMat
	            // dWc_level(n) = dWc

				if (lKid != -1) {
					val rKid = node(RKID)

					val ab  = kidsActs(lKid) << kidsActs(rKid)	// 2 * WORDSIZE X 1 (column vector)
					verbosePrint("					Calculating dWt", VERBOSE)
					val dWtbase = ab.toMat * ab.t.toMat   
					dWt_level(n) = pack3D( (0::WORDSIZE) {row => deltaDownFull(row) * dWtbase } )

					verbosePrint("					Calculating dW", VERBOSE)
					val dW = deltaDownFull.t.toMat * (ab.t << 1.0).toMat  // WORDSIZE X 1 * 1 X WORDSIZE 
					dW_level(n) = dW

					verbosePrint("					Calculating delta to children", VERBOSE)
					val linearPart = W.t * deltaDownFull.t
					val quadParts = (0::WORDSIZE) {row => 
										val Wt_row = get3D(Wt, 2*WORDSIZE, row)
										(Wt_row + Wt_row.t) * (deltaDownFull(row) * ab) 
									}
					val deltaDownBothVec = linearPart(0::2*WORDSIZE) + quadParts.sum

					// Pass on errors to each child
					deltas_level(lKid) = if (kidsLeafs(lKid) == 1 && WORDI) { deltaDownBothVec(0::WORDSIZE) }
										 else { deltaDownBothVec(0::WORDSIZE) * ab(0::WORDSIZE).map(d => 1 - d*d) }
					
					deltas_level(rKid) = if (kidsLeafs(rKid) == 1 && WORDI) { deltaDownBothVec(WORDSIZE::2*WORDSIZE) }
										 else { deltaDownBothVec(WORDSIZE::2*WORDSIZE) * ab(WORDSIZE::2*WORDSIZE).map(d => 1 - d*d) }
				}
				else {
					dWv_level(n - numParents) = (*, 0::Wv.numCols) {w => if (w == node(WORD)) deltaDownFull else DenseVector.zeros(WORDSIZE) }
				}
				(dWc)
			}
			// Sum deltas from level, add to total delta sum for each weight matrix
			dWc_tree += dWc_level.sum
			if (numParents > 0) {
				dW_tree += dW_level.sum 
				dWt_tree += dWt_level.sum
			}
			if (numLeaves > 0) {
				dWv_tree += dWv_level.sum
			}

			curLevel += 1
			(deltas_level)
		}	
		verbosePrint("			Completed backward propagation of tree", VERBOSE)
		(dWc_tree, dW_tree, dWt_tree, dWv_tree)
	}

	def main() = {
		val fanIn   = WORDSIZE.toDouble
		val range   = 1/sqrt(fanIn)
		val rangeW  = 1/sqrt(2*fanIn)
		val rangeWt = 1/sqrt(4*fanIn)
		val sizeWt  = WORDSIZE*2
		// NUMCLASSES X (WORDSIZE + 1)
		val Wc = (0::NUMCLASSES, 0::WORDSIZE + 1) {(row, col) => 
			if 		(col < WORDSIZE - 1) 	{ random[Double] * (2*range) - range }
			else if (col == WORDSIZE - 1)	{ random[Double] }
			else 							{ 0.0 }
		}
		
		// WORDSIZE X (WORDSIZE * 2 + 1)
		val W = (0::WORDSIZE, 0::(WORDSIZE*2 + 1)) {(row, col) => 
			if (col < WORDSIZE*2) {
				random[Double] * (2*rangeW) - rangeW
			}
			else { 0.0 }
		}

		// WORDSIZE X (WORDSIZE * 2) X (WORDSIZE * 2)
		val Wt = DenseMatrix.randn(WORDSIZE*sizeWt, sizeWt) * (2*rangeWt) - rangeWt

		// weights associated with individual phrases
		val Wv = DenseMatrix.randn(WORDSIZE, 3) * 0.1

		val tree = (0::2) { level =>
			if (level == 0) {
				(0::1, *) {row => tupleToDense6((-1, 0,  1, 2, 4, 2)) }
			}
			else {
				(0::2, *) {row =>
					if (row == 0) { tupleToDense6((0, -1, -1, 1, 4, 1)) }
					else 	     { tupleToDense6((0, -1, -1, 0, 2, 1)) }
				}
			}
		}
		Wv.pprint

		val (acts, outs, cost) = forwardPropTree(tree, Wc, W, Wt, Wv)
		val (dWc, dW, dWt, dWv) = backwardPropTree(tree, Wc, W, Wt, Wv, acts, outs)
		acts.pprint
		outs.pprint
		dWc.pprint
		//dW.pprint
		//dWt.pprint
		dWv.pprint
	}
} 