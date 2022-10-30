import random

def flattenList(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flattenList(S[0]) + flattenList(S[1:])
    return S[:1] + flattenList(S[1:])

class StateTreeBranch:
    def __init__(self, stateIndexes=[], parent=None, parentStateIndex=None, children=None):
        self.parent = parent
        self.parentStateIndex = parentStateIndex
        self.parentChildIndex = None
        self.stateIndexes = stateIndexes
        self.children = children
        if self.children is not None:
            for child in self.children:
                child.parent = self

    def addStateIndex(self, index):
        self.stateIndexes.append(index)

    def splitBranch(self, splitStateIndex):
        if splitStateIndex == self.stateIndexes[-1]:
            newChild = StateTreeBranch(parentStateIndex=splitStateIndex, stateIndexes=[], parent=self)
            if self.children is None:
                self.children = []
            self.children.append(newChild)

        elif splitStateIndex >= self.stateIndexes[0] and splitStateIndex < self.stateIndexes[-1]:
            childStateIndexes = []
            for idx in self.stateIndexes:
                if idx > splitStateIndex:
                    childStateIndexes.append(idx)

            for idx in childStateIndexes:
                self.stateIndexes.remove(idx)

            childrenOfNewChild = None
            if self.children is not None:
                for child in self.children:
                    if child.stateIndexes: # not empty
                        if child.stateIndexes[0] > splitStateIndex:
                            if childrenOfNewChild is None:
                                childrenOfNewChild = []
                            childrenOfNewChild.append(child)

            newChild1 = StateTreeBranch(parentStateIndex=splitStateIndex, stateIndexes=childStateIndexes, parent=self, children=childrenOfNewChild)
            newChild2 = StateTreeBranch(parentStateIndex=splitStateIndex, stateIndexes=[], parent=self)

            #if self.children is None:
            self.children = []

            self.children.append(newChild1)
            self.children.append(newChild2)
        else:
            print("ERROR: You should not be here!")
            print("splitStateIndex: " + str(splitStateIndex))
            print("self.stateIndexes: " + str(self.stateIndexes))
            print(self.printTree)

    def getBranchChildrenIndexes(self, maxDepth=None, excludeChildren=None):
        childrenIndexes = []

        children_ = self.children
        if children_: # not an empty list
            if excludeChildren: # not an empty list
                children_ = children_.copy()
                children_.remove(excludeChildren)
            for child in children_:
                L_ = len(child.stateIndexes)
                if maxDepth is None:
                    maxDepth_ = len(child.stateIndexes)
                    maxDepth2_ = None
                elif maxDepth > len(child.stateIndexes):
                    maxDepth_ = len(child.stateIndexes)
                    maxDepth2_ = maxDepth - maxDepth_
                else:
                    maxDepth_ = maxDepth
                    maxDepth2_ = maxDepth - maxDepth_

                tmp1 = []
                for i in range(maxDepth_):
                    tmp1.append(child.stateIndexes[i])
                if maxDepth2_ is not None:
                    if maxDepth2_ > 0:
                        tmp2 = child.getBranchChildrenIndexes(maxDepth=maxDepth2_)
                        if tmp2: # not empty
                            tmp1.append(tmp2[0])
                else:
                    tmp2 = child.getBranchChildrenIndexes(maxDepth=maxDepth2_)
                    if tmp2: # not empty
                        tmp1.append(tmp2[0])

                if tmp1:
                    childrenIndexes.append(tmp1[::-1])

        return childrenIndexes

    def getSiblingsStateIndexes(self):
        SiblingsStateIndexes = self.parent.getBranchChildrenIndexes(excludeChildren=self)
        return SiblingsStateIndexes

    def getStateIndexes(self, maxDepth=None):
        StateIndexes = self.stateIndexes[::-1]
        if maxDepth is None:
            StateIndexes_ = StateIndexes
        else:
            if maxDepth >= len(StateIndexes):
                StateIndexes_ = StateIndexes
            else:
                StateIndexes_ = StateIndexes[0:maxDepth]
        return StateIndexes_

        return AncestorsStateIndexes_

    def getAncestorsStateIndexes(self, maxDepth=None):
        AncestorsStateIndexes = self.stateIndexes[::-1]

        if self.parent is not None:
            AncestorsStateIndexes.extend(self.parent.getAncestorsStateIndexes())
        if maxDepth is None:
            AncestorsStateIndexes_ = AncestorsStateIndexes
        else:
            if maxDepth >= len(AncestorsStateIndexes):
                AncestorsStateIndexes_ = AncestorsStateIndexes
            else:
                AncestorsStateIndexes_ = AncestorsStateIndexes[0:maxDepth]

        return AncestorsStateIndexes_

    def getRelativesStateIndexes(self, maxDepth=None, excludeCurrentBranch=False):

        if not excludeCurrentBranch:
            RelativesStateIndexes_ = self.stateIndexes.copy()
        else:
            RelativesStateIndexes_ = []

        done = False
        L_ = len(RelativesStateIndexes_)
        if maxDepth is not None:
            if maxDepth <= L_:
                RelativesStateIndexes_ = RelativesStateIndexes_[L_-maxDepth:L_]
                done = True

        RelativesStateIndexes = []
        if not done:
            if maxDepth is not None:
                maxDepth_ = maxDepth-L_
            else:
                maxDepth_ = maxDepth

            if self.parent is not None:
                siblings = self.parent.children.copy()
                siblings.remove(self)
                if siblings: # not an empty list
                    for sibling in siblings:
                        siblingStateIndexes = sibling.stateIndexes.copy()[::-1]

                        L3_ = len(siblingStateIndexes)
                        if maxDepth_ is not None:
                            if maxDepth_ <= L3_:
                                siblingStateIndexes = siblingStateIndexes[L3_-(maxDepth_-1):L3_]
                        
                        if maxDepth_ is not None:
                            tmp = sibling.getBranchChildrenIndexes(maxDepth=maxDepth_-L3_-1)
                        else:
                            tmp = sibling.getBranchChildrenIndexes(maxDepth=maxDepth_)
                        if tmp:
                            siblingStateIndexes.insert(0, tmp)

                        if siblingStateIndexes:
                            RelativesStateIndexes.append(siblingStateIndexes)



            if self.parent is not None:
                parentStateIndexes = self.parent.getRelativesStateIndexes(maxDepth=maxDepth_)
                if parentStateIndexes:
                    L2_ = len(parentStateIndexes)
                    if maxDepth_ is None:
                        maxDepth2_ = L2_
                    else:
                        maxDepth2_ = maxDepth_
                    if maxDepth2_>1:
                        RelativesStateIndexes.append(parentStateIndexes[0:L2_-1])
                    RelativesStateIndexes_.insert(0, parentStateIndexes[-1])

        RelativesStateIndexes.extend(RelativesStateIndexes_)

        return RelativesStateIndexes

    def getBranchingPoints(self):
        if self.parent is None:
            BranchingPointsStateIndexes = []
            if self.stateIndexes: # not empty
                BranchingPointsStateIndexes.append(self.stateIndexes[0])
            BranchingPointsStateIndexes.extend(self.getBranchingPoints_())
            return BranchingPointsStateIndexes
        else:
            return self.parent.getBranchingPoints()

    def getBranchingPoints_(self):
        BranchingPointsStateIndexes = []
        if self.children is not None:
            BranchingPointsStateIndexes.append(self.stateIndexes[-1])
            for child in self.children:
                tmp = child.getBranchingPoints_()
                if tmp: # not empty
                    BranchingPointsStateIndexes.append(tmp)
        return BranchingPointsStateIndexes

    def getRandomPath(self, first=True):
        if self.parent is None:
            if first:
                forwardPaths = self.getAllChildBranches_()
                path = forwardPaths[random.randint(0,len(forwardPaths)-1)]
                return path
            else:
                return self.getAllChildBranches_()
        else:
            if first:
                backwardPath = self.getAncestorsStateIndexes()
                forwardPaths_ = self.parent.getRandomPath(first=False)

                if len(forwardPaths_) == 1:
                    return backwardPath
                else:
                    forwardPaths = forwardPaths_.copy()
                    forwardPaths.remove(backwardPath[::-1])
                    try:
                        forwardPath = forwardPaths[random.randint(0,len(forwardPaths)-1)]
                    except:
                        print("error in stateTree")
                        print("forwardPath = forwardPaths[random.randint(0,len(forwardPaths)-1)]")
                        print("State tree: ")
                        self.printTree()

                        print("backwardPath: " + str(backwardPath))
                        print("forwardPaths_: ")
                        for forwardPath in forwardPaths_:
                            print("     " + str(forwardPath))
                        print("forwardPaths")
                        for forwardPath in forwardPaths:
                            print("     " + str(forwardPath))

                        print("len(forwardPaths): " + str(len(forwardPaths)))
                        print("random.randint(0,len(forwardPaths)-1): " + str(random.randint(0,len(forwardPaths)-1)))

                    # print("backwardPath: " + str(backwardPath))
                    # print("forwardPaths_: ")
                    # for forwardPath in forwardPaths_:
                    #     print("     " + str(forwardPath))
                    # print("forwardPaths")
                    # for forwardPath in forwardPaths:
                    #     print("     " + str(forwardPath))


                    path = []
                    for stateIndexes in backwardPath:
                        if stateIndexes not in forwardPath:
                            path.append(stateIndexes)
                        else:  
                            path.append(stateIndexes)  # <-- might not be needed
                            break
                    for stateIndexes in forwardPath:
                        if stateIndexes not in backwardPath:
                            path.append(stateIndexes)          

                    return path
            else:
                return self.parent.getRandomPath(first=False)

    def getAllChildBranches_(self):
        if self.children is None: 
            return [self.getAncestorsStateIndexes()[::-1]]
        else:
            paths = []
            for child in self.children:
                paths_ = child.getAllChildBranches_()
                for path in paths_:
                    paths.append(path)

            return paths


    def printTree(self):
        if self.parent is None:
            allStateIndexes = []
            allStateIndexes.extend(self.printTree_())
            return allStateIndexes
        else:
            return self.parent.printTree()

    def printTree_(self, level=0):
        allStateIndexes = []
        allStateIndexes.extend(self.stateIndexes)

        indent = "  "
        indent = 2*indent
        print(level*indent + str(self.stateIndexes))
        if self.children is not None:
            for child in self.children:
                allStateIndexes.extend(child.printTree_(level=level+1))
        #else:
        #    print((level+1)*indent + str(None))

        return allStateIndexes

def example1():
    rootBranch = StateTreeBranch()
    currentBranch = rootBranch

    # state_reach
    for t in range(0,5+1):
        #print(t)
        currentBranch.addStateIndex(t)

    # impasse detected at 5
    t_impasse = t
    # backTracking
    for t in range(6,7+1):
        #print(t)
        ancestorsStateIndexes = currentBranch.getAncestorsStateIndexes()
        t_ = ancestorsStateIndexes[t-t_impasse]
        if t_ < currentBranch.stateIndexes[0]:
            currentBranch = currentBranch.parent


    # possible information gain
    currentBranch.splitBranch(t_)
    currentBranch = currentBranch.children[-1]
    for t in range(8,8+1):
        #print(t)
        currentBranch.addStateIndex(t)

    # enough progress from:
    #currentBranch.printTree()
    #print(currentBranch.getSiblingsStateIndexes())
    #print(currentBranch.parent.getBranchChildrenIndexes(excludeChildren=currentBranch))
    # state_reach
    for t in range(9,11+1):
        #print(t)
        currentBranch.addStateIndex(t)

    # impasse detected at 11
    t_impasse = t
    # backTracking
    for t in range(12,13+1):
        #print(t)
        ancestorsStateIndexes = currentBranch.getAncestorsStateIndexes()
        t_ = ancestorsStateIndexes[t-t_impasse]
        if t_ < currentBranch.stateIndexes[0]:
            currentBranch = currentBranch.parent


    # possible information gain
    currentBranch.splitBranch(t_)
    currentBranch = currentBranch.children[-1]
    for t in range(14,16+1):
        #print(t)
        currentBranch.addStateIndex(t)



    # possible information gain
    currentBranch.splitBranch(16)
    currentBranch = currentBranch.children[-1]
    for t in range(100,102+1):
        #print(t)
        currentBranch.addStateIndex(t)
    currentBranch.splitBranch(102)
    currentBranch = currentBranch.children[-1]
    for t in range(200,202+1):
        #print(t)
        currentBranch.addStateIndex(t)
    currentBranch = currentBranch.parent
    currentBranch = currentBranch.parent
    currentBranch.splitBranch(16)
    currentBranch = currentBranch.children[-1]
    for t in range(110,112+1):
        #print(t)
        currentBranch.addStateIndex(t)
    currentBranch = currentBranch.parent
    t = 16




    # backTracking
    t_impasse = t
    for t in range(17,19+1):
        #print(t)
        ancestorsStateIndexes = currentBranch.getAncestorsStateIndexes()
        t_ = ancestorsStateIndexes[t-t_impasse]
        if t_ < currentBranch.stateIndexes[0]:
            currentBranch = currentBranch.parent

    # possible information gain
    currentBranch.splitBranch(t_)
    currentBranch = currentBranch.children[-1]
    for t in range(19,21+1):
        #print(t)
        #if t == 20:
            #print(currentBranch.parent.getBranchChildrenIndexes())
        currentBranch.addStateIndex(t)

    t_ = 19
    currentBranch.splitBranch(t_)
    currentBranch = currentBranch.children[1]
    currentBranch.addStateIndex(200)

    #print(currentBranch.stateIndexes)
    #print(currentBranch.parent)
    currentBranch.printTree()

    print(currentBranch.getRandomPath())

    print(currentBranch.getAncestorsStateIndexes())

def example2():
    rootBranch = StateTreeBranch()
    currentBranch = rootBranch

    # state_reach
    for t in range(0,9):
        currentBranch.addStateIndex(t)

    currentBranch.splitBranch(4)
    currentBranch = currentBranch.children[1]

    for t in range(13,19):
        currentBranch.addStateIndex(t)

    currentBranch = currentBranch.parent
    currentBranch.splitBranch(3)
    currentBranch = currentBranch.children[1]
    currentBranch.addStateIndex(27)

    currentBranch = currentBranch.parent
    currentBranch.splitBranch(2)
    currentBranch = currentBranch.children[1]

    for t in range(30,36):
        currentBranch.addStateIndex(t)

    #currentBranch.printTree()
    #print(currentBranch.getRandomPath())

def example3():
    rootBranch = StateTreeBranch()
    currentBranch = rootBranch

    # state_reach
    for t in range(0,9+1):
        currentBranch.addStateIndex(t)

    currentBranch.splitBranch(9)
    currentBranch = currentBranch.children[0]

    for t in range(10,19):
        currentBranch.addStateIndex(t)

    currentBranch.printTree()
    print(currentBranch.getRandomPath())



def main():
    #example1()
    #example2()
    example3()


if __name__ == '__main__':
    main()
    #plt.show()