def compressData(inputData):
    '''
    parameters: inputData: pd.Dataframe
    return: inputData: pd.Dataframe
    Purpose: compress input data and resave according to type
    '''
    for eachType in set(inputData.dtypes.values):
        if 'int' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values:
                if inputData[i].min() < 0:
                    inputData[i] = pd.to_numeric(inputData[i],downcast='signed')
                else:
                    inputData[i] = pd.to_numeric(inputData[i],downcast='unsigned')      
        elif 'float' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values:   
                inputData[i] = pd.to_numeric(inputData[i],downcast='float')
        elif 'object' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values: 
                inputData[i] = inputData[i].astype('category')
    return inputData
 

# brand_id is populated with the mode of the brand_id corresponding to the seller_id
def get_Logs(filePath, userInfo):
    '''
    :parameters: None: None
    :return: userLog: pd.Dataframe
    :Purpose: 
    1.Convenient to retrieve raw behavioral data with other functions, while adjusting for missing provinces
    2.Use pickle module for sequence words, speed up reading and writing
    '''
    if os.path.exists(filePath):
        userLog = pickle.load(open(filePath,'rb'))
    else:
        userLog = pd.read_csv(userInfo, dtype=column_types)
        print('Is null? \n',userLog.isnull().sum())
 
        ## using mode to fill the missing values of brand_id 
        missingIndex = userLog[userLog.brand_id.isnull()].index
        sellerMode = userLog.groupby(['seller_id']).apply(lambda x:x.brand_id.mode()[0]).reset_index()
        pickUP = userLog.loc[missingIndex]
        pickUP = pd.merge(pickUP,sellerMode,how='left',on=['seller_id'])[0].astype('float32')
        pickUP.index = missingIndex
        userLog.loc[missingIndex,'brand_id'] = pickUP
        del pickUP, sellerMode, missingIndex
        print('--------------------')
        print('Is null? \n', userLog.isnull().sum())
        pickle.dump(userLog, open(filePath,'wb'))
    return userLog
    
userInfo = pd.read_csv('d:/JulyCompetition/input/user_log_format1.csv')
filePath = 'd:/JulyCompetition/features/Logs.pkl'
print('Before compressed:\n',userInfo.info())
userInfo = compressData(userInfo)
print('After compressed:\n',userInfo.info())
userLog = get_Logs(filePath,userInfo)
