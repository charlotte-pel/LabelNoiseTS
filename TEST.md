# Checking Data Generation
    # Launch python cmd checking generating data and noise:
        python gen_datasets.py
        
    # Checking noise generation:
        CheckNoiseFunc.checkNoiseForTwoFiveTenClass(True)
        
    # Checking data visualisation:
        generator = GenLabelNoiseTS(filename="dataset.h5", dir='pathToData' + 'Run' + str(1) + '/', csv=True,
                                    verbose=False)
        generator.visualisation(typePlot='mean')
        generator.visualisation(typePlot='mean', className='Corn')
        generator.visualisation(typePlot='all', className='Corn')
        generator.visualisation(typePlot='random', className='Corn', noProfile=20)
        generator.visualisation(typePlot='randomPoly', className='Corn')
        
# Checking Evaluation:
    pathTwoClass = './data/TwoClass/'
    pathFiveClass = './data/FiveClass/'
    pathTenClass = './data/TenClass/'
    
    # Checking Evaluation:
        EvalAlgo(path=pathTwoClass, noClass=2, seed=0, systematicChange=False)
        EvalAlgo(path=pathFiveClass, noClass=5, seed=0, systematicChange=False)
        EvalAlgo(path=pathFiveClass, noClass=5, seed=0, systematicChange=True)
        EvalAlgo(path=pathTenClass, noClass=10, seed=0, systematicChange=False)
        
    # Checking Evaluation Visualisation:
        visualisationEval('./results/evals/TwoClass/')
        visualisationEval('./results/evals/FiveClass/random')
        visualisationEval('./results/evals/FiveClass/systematic')
        visualisationEval('./results/evals/TenClass/')