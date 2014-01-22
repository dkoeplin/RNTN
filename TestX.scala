import optiml.compiler._
import optiml.library._
import optiml.shared._

object TestXInterpreter extends OptiMLApplicationInterpreter with TestX
object TestXCompiler extends OptiMLApplicationCompiler with TestX
trait TestX extends OptiMLApplication with Utilities{

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

	private def readTrees( filename: Rep[String] ): Rep[DenseVector[DenseVector[DenseMatrix[Int]]]] = {
		println("Loading trees...")
		val rawTrees = readMatrix(filename + "_trees.csv");
		println("Done.")
		
		//if (fullTreesRaw.numCols != TREECOLS) {
			// Assumes that data is saved such that parents are always below their children
			val roots    = rawTrees.getCol(PARENT).t.find(d => d == 0)
			val numTrees = roots.length
			println("Trees are unmodified. Pre-processing " + numTrees + " trees...")
			val treeFirstLeaves = (0::1).toDense << (roots + 1)

			(0::numTrees) { treeNum =>
				val curTree  = rawTrees.getRows(treeFirstLeaves(treeNum)::treeFirstLeaves(treeNum + 1))
				val numNodes = curTree.numRows
				val numLeafs = (numNodes + 1)/2

				val family = curTree.getCols(PARENT::LABEL).map(d => d.toInt - 1)
				val scores = curTree.getCol(LABEL)
				val levels = DenseVector[Int](numNodes, true)
				val leafs  = DenseVector[Int](numNodes, true)
				
				// Calculate level for each node
				var curNode = numNodes - 2
				while (curNode >= 0) {
					val parent = family(curNode, PARENT)
					val parentLevel = if (parent <= curNode || parent >= numNodes || parent < 0) {
	 					family.pprint
	 					errorOut("Error: There was a problem with the tree data: invalid parent given at node #" + curNode + " (requested node# " + parent + ")")
					}
					else { levels(parent) }
					levels(curNode) = parentLevel + 1
					curNode -= 1
				}
				
				// Calculate the number of leaves each node is an ancestor of (leaves are their own ancestors)
				curNode = 0
				while (curNode < numNodes) {
					leafs(curNode) = if (curNode < numLeafs) 1
								 	 else 					 leafs(family(curNode, LKID)) + leafs(family(curNode, RKID))
					curNode += 1
				}
				
				val numLevels = levels.max + 1
				val nodesAtLevels = (0::numLevels) {curLevel => 
					val nodesAtLevel = levels.find(l => l == curLevel) 
					val leafsAtLevel = leafs(nodesAtLevel)
					// within level, nodes are sorted from most to least leaf ancestors (leaves come last)
					val (leafsSorted, indices) = (-leafsAtLevel).sortWithIndex
					IndexVector(nodesAtLevel.toDense.apply(indices))
				}
				
				// Form tree levels, output tree
				(0::numLevels) { curLevel =>
					val parents  = if (curLevel == 0) { nodesAtLevels(0) } else { nodesAtLevels(curLevel - 1) }
					val children = if (curLevel == numLevels - 1) {nodesAtLevels(0)} else {nodesAtLevels(curLevel + 1)} 
					(nodesAtLevels(curLevel), *) {node =>
						(0::TREECOLS) {col =>
							if (col == PARENT) {
								val origParent = family(node, PARENT)
								if (origParent != -1) {parents.find(d => d == origParent).first } else { -1 }
							} 
							else if (col == LKID || col == RKID) { 
								val origChild = family(node, col)
								if (origChild != -1) { children.find(d => d == origChild).first } else { -1 }	
							}
							else if (col == WORD)   { family(node, WORD) }
							else if (col == LEAFS)  { leafs(node) }
							else { // (col == LABEL) 
								val score = scores(node)
								if (NUMCLASSES <= 3) {
									if 	  	( score <= 0.4 ) 	{ 0 } 
						 		  	else if ( score >= 0.6 ) 	{ NUMCLASSES - 1 }
								  	else if ( NUMCLASSES == 3 ) { 1 }
								  	else		   			    { POSNEGNEUT } // neutral
								}
								else {
									if (score > 1) { errorOut("Error: An invalid phrase score of " + score + " exists in the tree at node" + node +".") }
									else {floor(abs(score - 0.001)*NUMCLASSES) }
								}
							}
						}
					}
				}
			}
		/*}	TODO: FIX LOADING PREVIOUSLY FORMATTED TREES
		else {
			val roots    = fullTreesRaw.getCol(PARENT).t.find(d => d < 0)
			val numTrees = roots.length
			roots <<= fullTreesRaw.numRows
			(0::numTrees) { treeNum => 
				fullTreesRaw.getRows(roots(treeNum)::roots(treeNum + 1)).map(_.toInt)
			}
		}*/
	}

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

	def main = {
		val delim = ","
		println("Hello!")
		val DATASET = "/home/david/PPL/data/rotten"
		val MATRICES = "/home/david/PPL/outputs/"
		val trees = readTrees(DATASET)
		val Wc = readMatrix(MATRICES+"Wc_final.txt")
		val W  = readMatrix(MATRICES+"W_final.txt")
		val Wt = readMatrix(MATRICES+"Wt_final.txt")
		val Wv = readMatrix(MATRICES+"Wv_final.txt", delim)

		val classifications = (0::10) { n =>
			activateTree(trees(n), Wc, W, Wt, Wv)
		}
		classifications.t.pprint


	}

}