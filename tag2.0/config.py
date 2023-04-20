# SET VARIABLES
# Reinforcement learning model variables
class Config:
    def __init__(self,
                 maxEpsilon = 1, minEpsilon = 0, bootstrapValueEpsilon = 0.001, discountFactor = 0.99,
                 learningRate = 0.001, batchSize = 100, maxMemory = 5000,
                 gridSize = 5,
                 numPlayers = 2, numTaggers = 1,
                 renderSpeed = 0.3, createVideo = False,
                 numEpisodes = 10, numEpisodesBeforePrint = 10, minLoss = 1, numGamesShown = 100,
                 chanceOfMutation = 0.01,
                 savePath = r"C:\Users\Stanny\OneDrive - Trifork B.V\Documents\Tag/"
                ):
        
        # Model config
        self.maxEpsilon = maxEpsilon
        self.minEpsilon = minEpsilon
        self.bootstrapValueEpsilon = bootstrapValueEpsilon  # formerly lambda
        self.discountFactor = discountFactor  # formerly gamma
        self.learningRate = learningRate  # formerly alpha
        self.batchSize = batchSize
        self.maxMemory = maxMemory
        
        # Game config
        self.gridSize = gridSize
        self.numPlayers = numPlayers
        self.numTaggers = numTaggers
        self.numStates = 2 * numPlayers + 1
        self.numActions = 8
        
        # Match config
        self.numEpisodes = numEpisodes
        self.numEpisodesBeforePrint = numEpisodesBeforePrint
        self.minLoss = minLoss  # Stop when loss < minLoss
        
        # Render config
        self.renderSpeed = renderSpeed  # Players move every ~ seconds in rendering window
        self.numGamesShown = numGamesShown
        self.createVideo = createVideo
        
        # Evolution config
        self.chanceOfMutation = chanceOfMutation

        # Save path
        self.savePath = savePath
