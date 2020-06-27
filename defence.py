from captureAgents import CaptureAgent
from capture import SIGHT_RANGE
import random, time, util
from game import Directions,Actions
import game

class DefensiveInferenceAgent(CaptureAgent):
  """
  class for calculating sucessive position opponent pacman using bayesnet and elapse belief distribution ,simple greedy 
  ghost strategy
  """
  #####################
  # AI algorithm code #
  #####################

  depth=0
  
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    # Get all non-wall positions on the board
    self.legalPositions = gameState.data.layout.walls.asList(False)
    intialposition=gameState.getInitialAgentPosition(self.index)
    self.targetFoodLen=len(self.getFood(gameState).asList())
    self.maxdist_tofood=[]
    self.lastpositions=[]
    self.count=0
    for pos in self.getFoodYouAreDefending(gameState).asList():
      dist=util.manhattanDistance(pos,intialposition)
      if 18<=dist and dist<20:
        self.maxdist_tofood.append(pos)
    self.foodpos=random.choice(self.maxdist_tofood)
      

    # Initialize position belief distributions for opponents
    self.positionBeliefs = {}
    for opponent in self.getOpponents(gameState):
      self.initializeBeliefs(opponent)

  def initializeBeliefs(self, agent):
    """
    Uniformly initialize belief distributions for opponent positions.
    """
    self.positionBeliefs[agent] = util.Counter()
    for p in self.legalPositions:
      self.positionBeliefs[agent][p] = 1.0

  def chooseAction(self, gameState):
    # Update belief distribution about opponent positions and place hidden
    # opponents in their most likely positions
    myPosition = gameState.getAgentState(self.index).getPosition()
    noisyDistances = gameState.getAgentDistances()
    probableState = gameState.deepCopy()

    for opponent in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opponent)
      if pos:
        self.fixPosition(opponent, pos)
      else:
        self.elapseTime(opponent, gameState)
        self.observe(opponent, noisyDistances[opponent], gameState)

    self.displayDistributionsOverPositions(self.positionBeliefs.values())
    for opponent in self.getOpponents(gameState):
      probablePosition = self.guessPosition(opponent)
      conf = game.Configuration(probablePosition, Directions.STOP)
      probableState.data.agentStates[opponent] = game.AgentState(
        conf, probableState.isRed(probablePosition) != probableState.isOnRedTeam(opponent))
      
      
    actions = probableState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    values=[]
    bestVal= -10000.0
    b=0
    maxdistance=10000
    cnt=0
    best=None
    oppdist=util.Counter()
    
    for opp in self.getOpponents(probableState):
      if self.agentIsPacman(opp,probableState):
        cnt=cnt+1
      if gameState.getAgentState(opp).scaredTimer>0:
        oppdist[opp]=1
     
     
    if  cnt!=0 or cnt==0:
       x=self.getTeam(gameState)
       mindisofopp=10000.0
       
       for ind in x:
         if ind!=self.index and self.distancer.getDistance(myPosition,gameState.getAgentState(ind).getPosition())<=3:
           
           for opp in self.getOpponents(probableState):
             dist1=self.distancer.getDistance(self.guessPosition(opp),gameState.getAgentState(ind).getPosition())
             if mindisofopp>dist1:
               mindisofopp=dist1
               id=opp
       if mindisofopp<=3:
         maxdist2=0
         for action in actions:
          pos2=Actions.getSuccessor(myPosition,action)
          d=self.distancer.getDistance(self.guessPosition(id),pos2)
          if maxdist2<=d:
            maxdist2=d
            act=action
          
         return act
           
       
     
        
    for action in actions:
        for opponent in self.getOpponents(probableState):
            if ((not self.agentIsPacman(self.index,probableState)) and cnt==0)  or self.agentIsPacman(opponent,probableState):
                successor = self.getSuccessor(probableState, action)
                dist=self.minmax(opponent,successor,float("-inf"),float("inf"),self.depth,0)
                values.append(dist)
                if(bestVal<=dist):
                    bestVal=dist
                    best=action
                    pos=Actions.getSuccessor(myPosition,action)
                    b=1
        if self.agentIsPacman(self.index,probableState) and cnt==0 and probableState.getAgentState(self.index).numCarrying < 3 :        
           targetFood = self.getFood(gameState).asList()
           mypos=Actions.getSuccessor(myPosition,action)
           opponentDistances = self.getOpponentDistances(probableState)
           opponentDistance = min([dist for id, dist in opponentDistances]) 
           if oppdist[opponent]==1:
             opponentDistance=10000
           cappos=self.getCapsules(probableState)
           foodDistances = [self.distancer.getDistance(mypos, food)
                       for food in targetFood]
           minDistance = min(foodDistances) if len(foodDistances) else 1000
           dist1=self.distancer.getDistance(mypos,cappos[0]) if len(cappos) else 10000
           if dist1 < minDistance:
             minDistance=dist1
          
           if opponentDistance > minDistance+1 and minDistance<maxdistance:
              best=action
              maxdistance=minDistance
              pos=mypos
              b=1
              
            
                
                     
    if b==0:
        x=self.getFoodYouAreDefending(gameState).asList()
        bestVal=10000.0
        opponentDistances = self.getOpponentDistances(probableState)
        opponentDistance = min([dist for id, dist in opponentDistances]) 
        if len(x)>0:
            for action in actions:
                pos1=Actions.getSuccessor(myPosition,action)
                dist1=self.distancer.getDistance(self.foodpos ,pos1)
                values.append(dist1)
                if bestVal>=dist1 and opponentDistance >=1:
                  bestVal=dist1
                  best=action
                  
            if best!=None:
              return best
            else:
              return random.choice(actions)
            
            '''
            print(bestVal)
            bestActions = [a for a, v in zip(actions, values) if v == bestVal]
            return random.choice(bestActions)
            '''
        else:
            return random.choice(actions)
    '''
    if self.count>2 and len(self.lastpositions)>2 and self.lastpositions[-2]==pos:
      if len(actions)>1:
        actions.remove(best)
      act=random.choice(actions)
      pos2=Actions.getSuccessor(myPosition,act)
      self.lastpositions.append(pos2)
      self.count=0
      return act
    elif len(self.lastpositions)>2 and self.lastpositions[-2]==pos:
      self.count=self.count+1
      self.lastpositions.append(pos)
    else:
      self.lastpositions.append(pos)
      '''
  
    
      
    return best
                
  def fixPosition(self, agent, position):
    """
    Fix the position of an opponent in an agent's belief distributions.
    """
    updatedBeliefs = util.Counter()
    updatedBeliefs[position] = 1.0
    self.positionBeliefs[agent] = updatedBeliefs

  def elapseTime(self, agent, gameState):
    """
    Elapse belief distributions for an agent's position by one time step.
    Assume opponents move randomly, but also check for any food lost from
    the previous turn.
    """
    updatedBeliefs = util.Counter()
    for (oldX, oldY), oldProbability in self.positionBeliefs[agent].items():
      newDist = util.Counter()
      for p in [(oldX - 1, oldY), (oldX + 1, oldY),
                (oldX, oldY - 1), (oldX, oldY + 1)]:
        if p in self.legalPositions:
          newDist[p] = 1.0
      newDist.normalize()
      for newPosition, newProbability in newDist.items():
        updatedBeliefs[newPosition] += newProbability * oldProbability

    lastObserved = self.getPreviousObservation()
    if lastObserved:
      lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList()
                  if food not in self.getFoodYouAreDefending(gameState).asList()]
      for f in lostFood:
        updatedBeliefs[f] = 1.0/len(self.getOpponents(gameState))

    self.positionBeliefs[agent] = updatedBeliefs


  def observe(self, agent, noisyDistance, gameState):
    """
    Update belief distributions for an agent's position based upon
    a noisy distance measurement for that agent.
    """
    myPosition = self.getAgentPosition(self.index, gameState)
    teammatePositions = [self.getAgentPosition(teammate, gameState)
                         for teammate in self.getTeam(gameState)]
    updatedBeliefs = util.Counter()

    for p in self.legalPositions:
      if any([util.manhattanDistance(teammatePos, p) <= SIGHT_RANGE
              for teammatePos in teammatePositions]):
        updatedBeliefs[p] = 0.0
      else:
        trueDistance = util.manhattanDistance(myPosition, p)
        positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
        updatedBeliefs[p] = positionProbability * self.positionBeliefs[agent][p]

    if not updatedBeliefs.totalCount():
      self.initializeBeliefs(agent)
    else:
      updatedBeliefs.normalize()
      self.positionBeliefs[agent] = updatedBeliefs

  def guessPosition(self, agent):
    """
    Return the most likely position of the given agent in the game.
    """
    return self.positionBeliefs[agent].argMax()


  def getAgentPosition(self, agent, gameState):
    """
    Return the position of the given agent.
    """
    pos = gameState.getAgentPosition(agent)
    if pos:
      return pos
    else:
      return self.guessPosition(agent)

  def agentIsPacman(self, agent, gameState):
    """
    Check if the given agent is operating as a Pacman in its current position.
    """
    agentPos = self.getAgentPosition(agent, gameState)
    return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  
  def getOpponentDistances(self, gameState):
    """
    Return the IDs of and distances to opponents, relative to this agent.
    """
    return [(o, self.distancer.getDistance(
             self.getAgentPosition(self.index, gameState),
             self.getAgentPosition(o, gameState)))
            for o in self.getOpponents(gameState) ]
  
  
  
  def minmax(self,agent,gameState,alpha,beta,depth,sign):
    
    if (depth==0 and sign==0) or len(gameState.getLegalActions(self.index))==0:
      return -self.distancer.getDistance(self.getAgentPosition(self.index,gameState),self.getAgentPosition(agent,gameState))
    
    bestVal = float("-inf") if sign==0 else float("inf")
    if sign==0:
      actions=gameState.getLegalActions(self.index)
      actions.remove(Directions.STOP)
      for action in actions:
        successor=gameState.generateSuccessor(self.index,action)
        value=self.minmax(agent,successor,alpha,beta,depth-1,1)
        bestVal=max(value,bestVal)
        alpha=max(alpha,bestVal)
        if beta<=alpha:
          break
      return bestVal
    else:
      actions=gameState.getLegalActions(agent)
      actions.remove(Directions.STOP)
      for action in actions:
        successor=gameState.generateSuccessor(agent,action)
        value=self.minmax(agent,successor,alpha,beta,depth,0)
        bestVal=min(value,bestVal)
        beta=min(value,bestVal)
        if beta<=alpha:
          break
      return bestVal
      
      
    
 
  





