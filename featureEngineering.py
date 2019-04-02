# user basic information
userInfo = pd.read_csv('d:/JulyCompetition/input/user_info_format1.csv')
userInfo.age_range.fillna(userInfo.age_range.median(),inplace=True)#年龄用中位数填充
userInfo.gender.fillna(userInfo.gender.mode()[0],inplace=True)# 性别用众数填充
print('Check any missing value?\n',userInfo.isnull().any())# 检查缺省值
df_age = pd.get_dummies(userInfo.age_range,prefix='age')# 对age进行哑编码
df_sex = pd.get_dummies(userInfo.gender)# 对gender进行哑编码并改变列名
df_sex.rename(columns={0:'female',1:'male',2:'unknown'},inplace=True)
userInfo = pd.concat([userInfo.user_id, df_age, df_sex], axis=1)# 整合user信息
del df_age,df_sex
print(userInfo.info())
# 提取全部的原始行为数据...
totalActions = userLog[["user_id","action_type"]]
totalActions.head()
# 对行为类别进行哑编码，0 表示点击， 1 表示加入购物车, 2 表示购买，3 表示收藏.
df = pd.get_dummies(totalActions['action_type'],prefix='userTotalAction')

# 统计日志行为中用户点击、加购、购买、收藏的总次数
totalActions = pd.concat([totalActions.user_id, df], axis=1).groupby(['user_id'], as_index=False).sum()
totalActions['userTotalAction'] = totalActions['userTotalAction_0']+totalActions['userTotalAction_1']+totalActions['userTotalAction_2']+totalActions['userTotalAction_3']
del df
totalActions.info()
print('所有用户交互次数：'+str(userLog.shape[0]))
print('所有用户数：'+str(userLog['user_id'].nunique()))
print('所有用户平均交互次数：'+str(userLog.shape[0]/userLog['user_id'].nunique()))
totalActions['userTotalActionRatio'] = totalActions['userTotalAction']/userLog.shape[0]
totalActions['userTotalActionDiff'] = totalActions['userTotalAction']-userLog.shape[0]/userLog['user_id'].nunique()
print('所有用户点击次数：'+str(userLog[userLog.action_type==0].shape[0]))
totalActions['userClickRatio'] = totalActions['userTotalAction_0']/userLog[userLog.action_type==0].shape[0]
print('用户平均点击次数：'+str(userLog[userLog.action_type==0].shape[0]/userLog['user_id'].nunique()))
totalActions['userClickDiff'] = totalActions['userTotalAction_0']-userLog[userLog.action_type==0].shape[0]/userLog['user_id'].nunique()
print('所有用户加入购物车次数：'+str(userLog[userLog.action_type==1].shape[0]))
totalActions['userAddRatio'] = totalActions['userTotalAction_1']/userLog[userLog.action_type==1].shape[0]
print('用户平均加入购物车次数：'+str(userLog[userLog.action_type==1].shape[0]/userLog['user_id'].nunique()))
totalActions['userAddDiff'] = totalActions['userTotalAction_1']-userLog[userLog.action_type==1].shape[0]/userLog['user_id'].nunique()
print('所有用户购买次数：'+str(userLog[userLog.action_type==2].shape[0]))
totalActions['userBuyRatio'] = totalActions['userTotalAction_2']/userLog[userLog.action_type==2].shape[0]
print('用户平均购买次数：'+str(userLog[userLog.action_type==2].shape[0]/userLog['user_id'].nunique()))
totalActions['userBuyDiff'] = totalActions['userTotalAction_2']-userLog[userLog.action_type==2].shape[0]/userLog['user_id'].nunique()
print('所有用户收藏次数：'+str(userLog[userLog.action_type==3].shape[0]))
totalActions['userSaveRatio'] = totalActions['userTotalAction_3']/userLog[userLog.action_type==3].shape[0]
print('用户平均收藏次数：'+str(userLog[userLog.action_type==3].shape[0]/userLog['user_id'].nunique()))
totalActions['userSaveDiff'] = totalActions['userTotalAction_3']-userLog[userLog.action_type==3].shape[0]/userLog['user_id'].nunique()
# 统计用户点击，加购，收藏，购买次数占用户总交互次数的比例
totalActions['userClick_ratio'] = totalActions['userTotalAction_0']/totalActions['userTotalAction']
totalActions['userAdd_ratio'] = totalActions['userTotalAction_1']/totalActions['userTotalAction']
totalActions['userBuy_ratio'] = totalActions['userTotalAction_2']/totalActions['userTotalAction']
totalActions['userSave_ratio'] = totalActions['userTotalAction_3']/totalActions['userTotalAction']
# 统计日志行为中用户的点击、加购、收藏的购买转化率
totalActions['userTotalAction_0_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_0'])
totalActions['userTotalAction_0_ratio_diff'] = totalActions['userTotalAction_0_ratio'] - totalActions['userTotalAction_0_ratio'].mean()
totalActions['userTotalAction_1_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_1'])
totalActions['userTotalAction_1_ratio_diff'] = totalActions['userTotalAction_1_ratio'] - totalActions['userTotalAction_1_ratio'].mean()
totalActions['userTotalAction_3_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_3'])
totalActions['userTotalAction_3_ratio_diff'] = totalActions['userTotalAction_3_ratio'] - totalActions['userTotalAction_3_ratio'].mean()
totalActions.info()
days_cnt = userLog.groupby(['user_id'])['time_stamp'].nunique()
days_cnt_diff = days_cnt - userLog.groupby(['user_id'])['time_stamp'].nunique().mean()
# 对数值型特征手动标准化
numeric_cols = totalActions.columns[totalActions.dtypes == 'float64']
numeric_cols
numeric_col_means = totalActions.loc[:, numeric_cols].mean()
numeric_col_std = totalActions.loc[:, numeric_cols].std()
totalActions.loc[:, numeric_cols] = (totalActions.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
totalActions.head(5)
# 将统计好的数量和转化率进行拼接
userInfo = pd.merge(userInfo,totalActions,how='left',on=['user_id'])
del totalActions
userInfo.info()
# 用户六个月中做出行为的商品数量
item_cnt = userLog.groupby(['user_id'])['item_id'].nunique()
# 用户六个月中做出行为的种类数量
cate_cnt = userLog.groupby(['user_id'])['cat_id'].nunique()
# 用户六个月中做出行为的店铺数量
seller_cnt = userLog.groupby(['user_id'])['seller_id'].nunique()
# 用户六个月中做出行为的品牌数量
brand_cnt = userLog.groupby(['user_id'])['brand_id'].nunique()
# 用户六个月中做出行为的天数
days_cnt = userLog.groupby(['user_id'])['time_stamp'].nunique()

typeCount_result = pd.concat([item_cnt,cate_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,seller_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,brand_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,days_cnt],axis=1)
typeCount_result.rename(columns={'item_id':'item_cnt','cat_id':'cat_cnt','seller_id':'seller_cnt','brand_id':'brand_counts','time_stamp':'active_days'},inplace=True)
typeCount_result.reset_index(inplace=True)
typeCount_result.info()
# 对数值型特征手动标准化
numeric_cols = typeCount_result.columns[typeCount_result.dtypes == 'int64']
print(numeric_cols)
numeric_col_means = typeCount_result.loc[:, numeric_cols].mean()
numeric_col_std = typeCount_result.loc[:, numeric_cols].std()
typeCount_result.loc[:, numeric_cols] = (typeCount_result.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
typeCount_result.head(5)
## 将统计好的数量进行拼接
userInfo = pd.merge(userInfo,typeCount_result,how='left',on=['user_id'])
del typeCount_result
userInfo.info()
## 统计双十一之前，用户重复购买过的商家数量
### --------------------------------------------------------------------------
repeatSellerCount = userLog[["user_id","seller_id","time_stamp","action_type"]]
repeatSellerCount = repeatSellerCount[(repeatSellerCount.action_type == 2) & (repeatSellerCount.time_stamp < 1111)]
repeatSellerCount.drop_duplicates(inplace=True)
repeatSellerCount = repeatSellerCount.groupby(['user_id','seller_id'])['time_stamp'].count().reset_index()
repeatSellerCount = repeatSellerCount[repeatSellerCount.time_stamp > 1]
repeatSellerCount = repeatSellerCount.groupby(['user_id'])['seller_id'].count().reset_index()
repeatSellerCount.rename(columns={'seller_id':'repeat_seller_count'},inplace=True)
# 对数值型特征手动标准化
numeric_cols = repeatSellerCount.columns[repeatSellerCount.dtypes == 'int64']
print(numeric_cols)
numeric_col_means = repeatSellerCount.loc[:, numeric_cols].mean()
numeric_col_std = repeatSellerCount.loc[:, numeric_cols].std()
repeatSellerCount.loc[:, numeric_cols] = (repeatSellerCount.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
repeatSellerCount.head(5)
userInfo = pd.merge(userInfo,repeatSellerCount,how='left',on=['user_id'])
# 没有重复购买的user用0填充？
userInfo.repeat_seller_count.fillna(0,inplace=True)
userInfo['repeat_seller'] = userInfo['repeat_seller_count'].map(lambda x: 1 if x != 0 else 0)
del repeatSellerCount
# 用户总交互的次数、天数
# 用户交互的间隔
# 统计每月的点击次数，每月的加入购物次数，每月的购买次数，每月的收藏次数
### --------------------------------------------------------------------------
monthActionsCount = userLog[["user_id","time_stamp","action_type"]]
result = list()
for i in range(5,12):
    start = int(str(i)+'00')
    end = int(str(i)+'30')
    # 获取i月的数据
    example = monthActionsCount[(monthActionsCount.time_stamp >= start) & (monthActionsCount.time_stamp < end)]
    # 对i月的交互行为进行哑编码
    df = pd.get_dummies(example['action_type'],prefix='%d_Action'%i)
    df[str(i)+'_Action'] = df[str(i)+'_Action_0']+df[str(i)+'_Action_1']+df[str(i)+'_Action_2']+df[str(i)+'_Action_3']
    # 将example的time_stamp设为月份值（5,6，。。。，11）
    example.loc[:,'time_stamp'] = example.time_stamp.apply(lambda x: int(str(x)[0]) if len(str(x)) == 3 else int(str(x)[:2]))
    result.append(pd.concat([example, df], axis=1).groupby(['user_id','time_stamp'],as_index=False).sum())

for i in range(0,7):
    userInfo = pd.merge(userInfo,result[i],how='left',on=['user_id'])
    userInfo.fillna(0,inplace=True)
for col in ['time_stamp_x','action_type_x','time_stamp_y','action_type_y','time_stamp','action_type']:
    del userInfo[col]
for i in range(5,12):
    userInfo[str(i)+'_Action'] = userInfo[str(i)+'_Action_0']+userInfo[str(i)+'_Action_1']+userInfo[str(i)+'_Action_2']+userInfo[str(i)+'_Action_3']
filePath='d:/JulyCompetition/features/userInfo_Features.pkl'
pickle.dump(userInfo, open(filePath, 'wb'))
# 读取用户特征
filePath='d:/JulyCompetition/features/userInfo_Features.pkl'
if os.path.exists(filePath):
    userInfo = pickle.load(open(filePath,'rb'))
userInfo.info()
# 统计每个商户的商品，种类，品牌总数，并放入dataFrame[seller_id,xx_number]为列名，便于往后的拼接
# （表示商户的规模大小）
itemNumber = userLog[['seller_id','item_id']].groupby(['seller_id'])['item_id'].nunique().reset_index()
catNumber = userLog[['seller_id','cat_id']].groupby(['seller_id'])['cat_id'].nunique().reset_index()
brandNumber = userLog[['seller_id','brand_id']].groupby(['seller_id'])['brand_id'].nunique().reset_index()
itemNumber.rename(columns={'item_id':'item_number'},inplace=True)
catNumber.rename(columns={'cat_id':'cat_number'},inplace=True)
brandNumber.rename(columns={'brand_id':'brand_number'},inplace=True)
 # 统计商户重复买家总数量（表示商户对于新用户的留存能力）
repeatPeoCount = userLog[(userLog.time_stamp < 1111) & (userLog.action_type == 2)]
repeatPeoCount = repeatPeoCount.groupby(['seller_id'])['user_id'].value_counts().to_frame()
repeatPeoCount.rename(columns={'user_id':'Buy_Number'},inplace=True)
repeatPeoCount.reset_index(inplace=True)
repeatPeoCount = repeatPeoCount[repeatPeoCount.Buy_Number > 1]
repeatPeoCount = repeatPeoCount.groupby(['seller_id']).apply(lambda x:len(x.user_id)).reset_index()
repeatPeoCount = pd.merge(pd.DataFrame({'seller_id':range(1, 4996 ,1)}),repeatPeoCount,how='left',on=['seller_id']).fillna(0)
repeatPeoCount.rename(columns={0:'repeatBuy_peopleNumber'},inplace=True)
##统计被点击，被加入购物车，被购买，被收藏次数
###统计被点击购买转化率，被加入购物车购买转化率，被收藏次数购买转化率
sellers = userLog[["seller_id","action_type"]]
df = pd.get_dummies(sellers['action_type'],prefix='seller')
sellers = pd.concat([sellers, df], axis=1).groupby(['seller_id'], as_index=False).sum()
sellers.drop("action_type", axis=1,inplace=True)
del df
#　构造转化率字段
sellers['seller_0_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_0'])
sellers['seller_1_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_1'])
sellers['seller_3_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_3'])
sellers.info()
###统计每个商户被点击的人数，被加入购物车的人数，被购买的人数，被收藏的人数
peoCount = userLog[["user_id","seller_id","action_type"]]
df = pd.get_dummies(peoCount['action_type'],prefix='seller_peopleNumber')
peoCount = pd.concat([peoCount, df], axis=1)
peoCount.drop("action_type", axis=1,inplace=True)
peoCount.drop_duplicates(inplace=True)
df1 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_0.sum())
df2 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_1.sum())
df3 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_2.sum())
df4 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_3.sum())
peoCount = pd.concat([df1, df2,df3, df4], axis=1).reset_index()
del df1,df2,df3,df4
peoCount.rename(columns={0:'seller_peopleNum_0',1:'seller_peopleNum_1',2:'seller_peopleNum_2',3:'seller_peopleNum_3'},inplace=True)
peoCount.info()
###对各种统计表根据seller_id进行拼接
sellers = pd.merge(sellers,peoCount,on=['seller_id'])
sellers = pd.merge(sellers,itemNumber,on=['seller_id'])
sellers = pd.merge(sellers,catNumber,on=['seller_id'])
sellers = pd.merge(sellers,brandNumber,on=['seller_id'])
sellers = pd.merge(sellers,repeatPeoCount,on=['seller_id'])
del itemNumber,catNumber,brandNumber,peoCount,repeatPeoCount
sellers.info()
# 统计每个商户的商品数，商品种类、品牌占总量的比例（表示商户的规模大小）
sellers['item_ratio'] = sellers['item_number']/userLog['item_id'].nunique()
sellers['cat_ratio'] = sellers['item_number']/userLog['cat_id'].nunique()
sellers['brand_ratio'] = sellers['item_number']/userLog['brand_id'].nunique()
# 统计每个商户被点击、加购、购买、收藏的人数占有点击、加购、购买、收藏行为人数的比例
sellers['click_people_ratio'] = sellers['seller_peopleNum_0']/userLog[userLog['action_type'] == 0]['user_id'].nunique()
sellers['add_people_ratio'] = sellers['seller_peopleNum_1']/userLog[userLog['action_type'] == 1]['user_id'].nunique()
sellers['buy_people_ratio'] = sellers['seller_peopleNum_2']/userLog[userLog['action_type'] == 2]['user_id'].nunique()
sellers['save_people_ratio'] = sellers['seller_peopleNum_3']/userLog[userLog['action_type'] == 3]['user_id'].nunique()
# 对数值型特征手动标准化
numeric_cols = sellers.columns[sellers.dtypes != 'uint64']
print(numeric_cols)
numeric_col_means = sellers.loc[:, numeric_cols].mean()
numeric_col_std = sellers.loc[:, numeric_cols].std()
sellers.loc[:, numeric_cols] = (sellers.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
sellers.head(5)
filePath='d:/JulyCompetition/features/sellerInfo_Features.pkl'
pickle.dump(sellers,open(filePath,'wb'))
# 读取商户特征
filePath='d:/JulyCompetition/features/sellerInfo_Features.pkl'
if os.path.exists(filePath):
    sellers = pickle.load(open(filePath,'rb'))
## 提取预测目标的行为数据
trainData = pd.read_csv('d:/JulyCompetition/input/train_format1.csv')
trainData.rename(columns={'merchant_id':'seller_id'},inplace=True)
testData = pd.read_csv('d:/JulyCompetition/input/test_format1.csv')
testData.rename(columns={'merchant_id':'seller_id'},inplace=True)
targetIndex = pd.concat([trainData[['user_id', 'seller_id']],testData[['user_id', 'seller_id']]],ignore_index=True)
logs = pd.merge(targetIndex,userLog,on=['user_id', 'seller_id'])
del trainData,testData,targetIndex
logs.info()
### 统计用户对预测的商店的行为特征，例如点击，加入购物车，购买，收藏的总次数,以及各种转化率
df_result = logs[["user_id", "seller_id","action_type"]]
df = pd.get_dummies(df_result['action_type'],prefix='userSellerAction')
df_result = pd.concat([df_result, df], axis=1).groupby(['user_id', 'seller_id'], as_index=False).sum()
del df
df_result.drop("action_type", axis=1,inplace=True)
df_result['userSellerAction_0_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_0'])
df_result['userSellerAction_1_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_1'])
df_result['userSellerAction_3_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_3'])
df_result.info()
###统计用户对预测商店点击的总天数
clickDays = logs[logs.action_type == 0]
clickDays = clickDays[["user_id", "seller_id","time_stamp","action_type"]]
clickDays = clickDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
clickDays.rename(columns={0:'click_days'},inplace=True)
df_result = pd.merge(df_result,clickDays,how='left',on=['user_id', 'seller_id'])
df_result.click_days.fillna(0,inplace=True)
del clickDays
###统计用户对预测商店加入购物车的总天数
addDays = logs[logs.action_type == 1]
addDays = addDays[["user_id", "seller_id","time_stamp","action_type"]]
addDays = addDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
addDays.rename(columns={0:'add_days'},inplace=True)
df_result = pd.merge(df_result,addDays,how='left',on=['user_id', 'seller_id'])
df_result.add_days.fillna(0,inplace=True)
del addDays
###统计用户对预测商店购物的总天数
buyDays = logs[logs.action_type == 2]
buyDays = buyDays[["user_id", "seller_id","time_stamp","action_type"]]
buyDays = buyDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
buyDays.rename(columns={0:'buy_days'},inplace=True)
df_result = pd.merge(df_result,buyDays,how='left',on=['user_id', 'seller_id'])
df_result.buy_days.fillna(0,inplace=True)
del buyDays
###统计用户对预测商店购物的总天数
saveDays = logs[logs.action_type == 3]
saveDays = saveDays[["user_id", "seller_id","time_stamp","action_type"]]
saveDays = saveDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
saveDays.rename(columns={0:'save_days'},inplace=True)
df_result = pd.merge(df_result,saveDays,how='left',on=['user_id', 'seller_id'])
df_result.save_days.fillna(0,inplace=True)
del saveDays
itemCount = logs[["user_id", "seller_id","item_id","action_type"]]
# 点击商品数量
itemCountClick = itemCount[itemCount.action_type == 0]
item_result = itemCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_click_count'},inplace=True)
item_result.item_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountClick,item_result
# 加入购物车商品数量
itemCountAdd = itemCount[itemCount.action_type == 1]
item_result = itemCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_add_count'},inplace=True)
item_result.item_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountAdd,item_result
# 购买商品数量
itemCountBuy = itemCount[itemCount.action_type == 2]
item_result = itemCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_buy_count'},inplace=True)
item_result.item_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountBuy,item_result
# 收藏商品数量
itemCountSave = itemCount[itemCount.action_type == 3]
item_result = itemCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_save_count'},inplace=True)
item_result.item_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountSave,item_result
catCount = logs[["user_id", "seller_id","cat_id","action_type"]]
# 点击种类数量
catCountClick = catCount[catCount.action_type == 0]
cat_result = catCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_click_count'},inplace=True)
cat_result.cat_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountClick,cat_result
# 加入购物车种类数量
catCountAdd = catCount[catCount.action_type == 1]
cat_result = catCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_add_count'},inplace=True)
cat_result.cat_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountAdd,cat_result
# 购买种类数量
catCountBuy = catCount[catCount.action_type == 2]
cat_result = catCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_buy_count'},inplace=True)
cat_result.cat_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountBuy,cat_result
# 收藏种类数量
catCountSave = catCount[catCount.action_type == 3]
cat_result = catCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_save_count'},inplace=True)
cat_result.cat_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountSave,cat_result
brandCount = logs[["user_id", "seller_id","brand_id","action_type"]]
# 点击品牌数量
brandCountClick = brandCount[brandCount.action_type == 0]
brand_result = brandCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_click_count'},inplace=True)
brand_result.brand_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountClick,brand_result
# 加入购物车品牌数量
brandCountAdd = brandCount[brandCount.action_type == 1]
brand_result = brandCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_add_count'},inplace=True)
brand_result.brand_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountAdd,brand_result
# 购买品牌数量
brandCountBuy = brandCount[brandCount.action_type == 2]
brand_result = brandCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_buy_count'},inplace=True)
brand_result.brand_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountBuy,brand_result
# 收藏品牌数量
brandCountSave = brandCount[brandCount.action_type == 3]
brand_result = brandCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_save_count'},inplace=True)
brand_result.brand_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountSave,brand_result
df_result.fillna(0,inplace=True)
# 对数值型特征手动标准化
for col in ['buy_days','item_buy_count','cat_buy_count','brand_buy_count']:
    df_result[col] = df_result[col].astype('float64')
# 对数值型特征手动标准化
numeric_cols = df_result.columns[df_result.dtypes == 'float64']
print(numeric_cols)
numeric_col_means = df_result.loc[:, numeric_cols].mean()
numeric_col_std = df_result.loc[:, numeric_cols].std()
df_result.loc[:, numeric_cols] = (df_result.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
df_result.head(5)
filePath='d:/JulyCompetition/features/userSellerActions.pkl'
pickle.dump(df_result,open(filePath,'wb'))
# 读取商户特征
filePath='d:/JulyCompetition/features/userSellerActions.pkl'
if os.path.exists(filePath):
    df_results = pickle.load(open(filePath,'rb'))
