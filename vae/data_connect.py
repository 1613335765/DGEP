import pandas as pd

# 读取csv文件
df1 = pd.read_csv('C:\\Users\\16133\\Desktop\\python\\datasets\\class_labels.csv')
df2 = pd.read_csv('C:\\Users\\16133\\Desktop\\python\\datasets\\features_epoch=150_ycc=3.csv')
# df3 = pd.read_csv('C:\\Users\\QXC44\\Desktop\\dachaung\\daima\\GenesRelateDiseases\\datasets\\gtex_features.csv',usecols=lambda col: col not in ['sum_exp','total'])
# df4 = pd.read_csv('C:\\Users\\QXC44\\Desktop\\dachaung\\daima\\GenesRelateDiseases\\datasets\\kegg_features.csv')
# df5 = pd.read_csv('C:\\Users\\QXC44\\Desktop\\dachaung\\daima\\GenesRelateDiseases\\datasets\\pathdipall_features.csv')
# df6 = pd.read_csv('C:\\Users\\QXC44\\Desktop\\dachaung\\daima\\GenesRelateDiseases\\datasets\\ppi_features.csv',usecols=lambda col: col not in ['ppi_total'])

# df1=df1.dropna()
# df2=df2.dropna()
# df3=df3.dropna()
# df3=df3.iloc[:, :61]
# df4=df4.dropna()
# df5=df5.dropna()
# df6=df6.dropna()

# 执行自然连接操作
result = pd.merge(df1, df2, how='inner', on='entrezId')
# result = pd.merge(result, df3, how='inner', on='entrezId')
# result = pd.merge(result, df4, how='inner', on='entrezId')
# result = pd.merge(result, df5, how='inner', on='entrezId')
# result = pd.merge(result, df6, how='inner', on='entrezId')

# 将结果保存到zonghe.csv文件中
result.to_csv('C:\\Users\\16133\\Desktop\\python\\datasets\\connected_data.csv', index=False)
