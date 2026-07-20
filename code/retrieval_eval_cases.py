"""Corpus-grounded retrieval evaluation case data.

Rows use: name, user query, retrieval query, expected topic, exact expected path.
The primary rows are frozen independently of retrieval candidate outputs.
"""

PRIMARY_CASE_ROWS = (
    # 00-Prerequisites/LinearAlgebra.md
    ("linear_algebra", "线性代数主要研究什么？", "linear algebra equations vectors matrices fundamentals", "Prerequisites", "00-Prerequisites/LinearAlgebra.md"),
    ("linear_algebra_dot_outer_product", "向量点积和外积有什么区别？", "vector dot product outer product multiplication", "Prerequisites", "00-Prerequisites/LinearAlgebra.md"),
    ("linear_algebra_matrix_multiplication", "矩阵乘法是怎么定义的？", "matrix multiplication dot product linear algebra", "Prerequisites", "00-Prerequisites/LinearAlgebra.md"),

    # 00-Prerequisites/README.md
    ("prerequisites_overview", "学习机器学习前需要哪些基础？", "machine learning prerequisites linear algebra statistics types of data", "Prerequisites", "00-Prerequisites/README.md"),
    ("prerequisites_math_topics", "机器学习先修数学包含哪些主题？", "prerequisite mathematics linear algebra statistics", "Prerequisites", "00-Prerequisites/README.md"),
    ("prerequisites_data_foundations", "先修章节里哪里介绍数据类型？", "prerequisites types of data chapter index", "Prerequisites", "00-Prerequisites/README.md"),

    # 00-Prerequisites/Statistics.md
    ("statistics", "统计学基础包括哪些内容？", "statistics basics probabilities sampling confidence intervals hypothesis testing", "Prerequisites", "00-Prerequisites/Statistics.md"),
    ("statistics_probability_types", "概率有哪些常见类型？", "statistics probability types", "Prerequisites", "00-Prerequisites/Statistics.md"),
    ("statistics_confidence_intervals", "置信区间是什么？", "statistics confidence intervals sampling", "Prerequisites", "00-Prerequisites/Statistics.md"),
    ("statistics_hypothesis_testing", "统计假设检验如何理解？", "statistics hypothesis testing null alternative", "Prerequisites", "00-Prerequisites/Statistics.md"),

    # 00-Prerequisites/TypesOfData.md
    ("types_of_data", "数据有哪些基本类型？", "types of data cross-sectional time series", "Prerequisites", "00-Prerequisites/TypesOfData.md"),
    ("cross_sectional_data", "什么是横截面数据？", "cross-sectional data observations point in time", "Prerequisites", "00-Prerequisites/TypesOfData.md"),
    ("time_series_data_type", "什么是时间序列数据？", "time series data observations over time data type", "Prerequisites", "00-Prerequisites/TypesOfData.md"),

    # 01-Regression/00-RegressionModelsComparison.md
    ("regression_comparison", "回归模型如何比较表现？", "regression model performance evaluation R-squared adjusted R-squared", "Regression", "01-Regression/00-RegressionModelsComparison.md"),
    ("regression_r_squared", "R-squared 衡量了什么？", "R-squared coefficient determination regression performance", "Regression", "01-Regression/00-RegressionModelsComparison.md"),
    ("regression_adjusted_r_squared", "为什么加入变量后要看 adjusted R-squared？", "adjusted R-squared variables penalty regression comparison", "Regression", "01-Regression/00-RegressionModelsComparison.md"),

    # 01-Regression/01-LinearRegression.md
    ("linear_regression_variable_selection", "线性回归怎么做变量选择？", "linear regression variable selection feature selection p-value backward elimination", "Regression", "01-Regression/01-LinearRegression.md"),
    ("linear_regression_assumptions", "线性回归模型依赖哪些假设？", "linear regression assumptions errors linearity homoscedasticity independence", "Regression", "01-Regression/01-LinearRegression.md"),
    ("dummy_variable_trap", "dummy variable trap 是什么？", "dummy variable trap categorical variables linear regression", "Regression", "01-Regression/01-LinearRegression.md"),
    ("linear_regression_mse", "线性回归如何用均方误差衡量准确性？", "linear regression mean squared error measure accuracy", "Regression", "01-Regression/01-LinearRegression.md"),
    ("backward_elimination_code", "backward elimination 用哪个函数实现？", "backward elimination statsmodels OLS add constant code", "Regression", "01-Regression/01-LinearRegression.md"),

    # 01-Regression/02-PolynomialRegression.md
    ("polynomial_regression", "多项式回归是什么？", "polynomial linear regression nonlinear relationship", "Regression", "01-Regression/02-PolynomialRegression.md"),
    ("polynomial_features_degree", "多项式回归怎样生成不同次数的特征？", "polynomial features degree fit_transform X_poly", "Regression", "01-Regression/02-PolynomialRegression.md"),
    ("polynomial_regression_python", "Python 里如何拟合多项式回归？", "Python PolynomialFeatures LinearRegression fit polynomial", "Regression", "01-Regression/02-PolynomialRegression.md"),

    # 01-Regression/03-SupportVectorRegression.md
    ("support_vector_regression", "支持向量回归的基本原理是什么？", "support vector regression SVR basic principle epsilon tube", "Regression", "01-Regression/03-SupportVectorRegression.md"),
    ("svr_nonlinear_preprocessing", "SVR 如何通过预处理处理非线性？", "support vector regression nonlinearity preprocessing feature transformation", "Regression", "01-Regression/03-SupportVectorRegression.md"),
    ("svr_kernel_mapping", "SVR 的核函数怎样实现隐式映射？", "SVR kernels implicit mapping nonlinear", "Regression", "01-Regression/03-SupportVectorRegression.md"),

    # 01-Regression/README.md
    ("regression_overview", "什么是回归问题？", "regression predict dependent variable continuous numeric domain", "Regression", "01-Regression/README.md"),
    ("regression_error_assumptions", "回归误差需要满足哪些分布和方差条件？", "regression errors normal independent zero mean constant variance homoscedasticity", "Regression", "01-Regression/README.md"),
    ("regression_model_families", "回归章节包含哪些模型？", "regression linear polynomial support vector regression models", "Regression", "01-Regression/README.md"),

    # 02-Classification/04-NaiveBayes.md
    ("naive_bayes", "什么是朴素贝叶斯分类器？", "naive bayes classifier Bayes theorem probability", "Classification", "02-Classification/04-NaiveBayes.md"),
    ("naive_bayes_procedure", "朴素贝叶斯分类算法有哪些步骤？", "naive bayes classifier algorithm procedure", "Classification", "02-Classification/04-NaiveBayes.md"),
    ("naive_bayes_types", "朴素贝叶斯有哪些类型？", "types of naive bayes classifiers Gaussian multinomial Bernoulli", "Classification", "02-Classification/04-NaiveBayes.md"),
    ("naive_bayes_tradeoffs", "朴素贝叶斯有哪些优点和缺点？", "naive bayes advantages disadvantages", "Classification", "02-Classification/04-NaiveBayes.md"),

    # 02-Classification/README.md
    ("classification_overview", "什么是机器学习分类问题？", "classification supervised machine learning discrete classes", "Classification", "02-Classification/README.md"),
    ("classification_methods", "分类有哪些常见方法？", "classification algorithms logistic regression KNN SVM naive bayes decision tree random forest", "Classification", "02-Classification/README.md"),
    ("classification_algorithm_index", "分类章节列出了哪些分类器？", "types of classification algorithms chapter index", "Classification", "02-Classification/README.md"),

    # 03-Clustering/01-K-meansClustering.md
    ("kmeans_steps", "K-means 聚类的步骤是什么？", "K-means clustering algorithm steps centroids assignment", "Clustering", "03-Clustering/01-K-meansClustering.md"),
    ("kmeans_objective", "K-means 优化的目标函数是什么？", "K-means clustering objective function minimize distance", "Clustering", "03-Clustering/01-K-meansClustering.md"),
    ("soft_kmeans", "Soft K-means 和普通 K-means 有什么不同？", "soft K-means algorithm responsibilities objective", "Clustering", "03-Clustering/01-K-meansClustering.md"),
    ("kmeans_centroid_assignment", "K-means 如何分配样本并更新质心？", "K-means assign points clusters update centroids", "Clustering", "03-Clustering/01-K-meansClustering.md"),

    # 03-Clustering/02-HierarchicalClustering.md
    ("hierarchical_clustering", "层次聚类如何逐步合并簇？", "hierarchical clustering merge closest clusters linkage", "Clustering", "03-Clustering/02-HierarchicalClustering.md"),
    ("hierarchical_dendrogram", "层次聚类中的 dendrogram 怎么看？", "hierarchical clustering dendrogram tree", "Clustering", "03-Clustering/02-HierarchicalClustering.md"),
    ("hierarchical_dissimilarity_threshold", "如何用不相似度阈值确定层次聚类数量？", "hierarchical clustering dissimilarity threshold number clusters", "Clustering", "03-Clustering/02-HierarchicalClustering.md"),

    # 03-Clustering/README.md
    ("clustering_overview", "无标签数据为什么需要聚类？", "clustering unsupervised learning unlabeled data groupings", "Clustering", "03-Clustering/README.md"),
    ("clustering_density_estimation", "密度估计在无监督学习中做什么？", "density estimation probability density unsupervised learning samples", "Clustering", "03-Clustering/README.md"),
    ("clustering_latent_variables", "无监督学习如何发现潜变量？", "latent variables underlying cause unsupervised learning", "Clustering", "03-Clustering/README.md"),

    # 04-AssociationRuleLearning/01-Apriori.md
    ("apriori", "Apriori 算法是什么？", "Apriori association rule learning algorithm", "Association Rule Learning", "04-AssociationRuleLearning/01-Apriori.md"),
    ("apriori_metrics", "Apriori 中 support、confidence 和 lift 分别表示什么？", "Apriori support confidence lift association rules", "Association Rule Learning", "04-AssociationRuleLearning/01-Apriori.md"),
    ("apriori_steps", "Apriori 如何筛选频繁项集和关联规则？", "Apriori algorithm steps minimum support confidence frequent itemsets", "Association Rule Learning", "04-AssociationRuleLearning/01-Apriori.md"),

    # 04-AssociationRuleLearning/02-Eclat.md
    ("eclat", "Eclat 模型如何衡量项集的普遍程度？", "Eclat model support prevalence set of items", "Association Rule Learning", "04-AssociationRuleLearning/02-Eclat.md"),
    ("eclat_support_only", "为什么 Eclat 主要使用 support？", "Eclat support prevalence sets of items", "Association Rule Learning", "04-AssociationRuleLearning/02-Eclat.md"),
    ("eclat_steps", "Eclat 如何按支持度筛选和排序事务子集？", "Eclat minimum support select subsets sort decreasing support", "Association Rule Learning", "04-AssociationRuleLearning/02-Eclat.md"),

    # 04-AssociationRuleLearning/README.md
    ("association_overview", "什么是关联规则学习？", "association rule learning unsupervised hidden rules sets", "Association Rule Learning", "04-AssociationRuleLearning/README.md"),
    ("association_hidden_rules", "如何从一组物品中发现不明显的规则？", "find not apparent rules sets association rule learning", "Association Rule Learning", "04-AssociationRuleLearning/README.md"),
    ("association_algorithms", "关联规则章节包含哪些算法？", "association rule learning Apriori Eclat algorithms", "Association Rule Learning", "04-AssociationRuleLearning/README.md"),

    # 06-NaturalLanguageProcessing/README.md
    ("nlp_overview", "自然语言处理主要解决什么问题？", "natural language processing NLP models use cases", "Natural Language Processing", "06-NaturalLanguageProcessing/README.md"),
    ("nlp_bag_of_words", "Bag of Words 模型如何预处理文本？", "bag of words preprocessing procedure NLP", "Natural Language Processing", "06-NaturalLanguageProcessing/README.md"),
    ("nlp_named_entity_recognition", "命名实体识别属于哪类 NLP 应用？", "named entity recognition NER NLP use case", "Natural Language Processing", "06-NaturalLanguageProcessing/README.md"),
    ("nlp_tfidf_stemming", "TF-IDF、stemming 和 lemmatization 分别是什么？", "TF-IDF stemming lemmatization NLP jargon", "Natural Language Processing", "06-NaturalLanguageProcessing/README.md"),

    # 07-DeepLearning/README.md
    ("deep_learning_overview", "神经网络是怎样工作的？", "neural networks deep learning working procedure", "Deep Learning", "07-DeepLearning/README.md"),
    ("neural_network_components", "神经元、层和突触在神经网络中分别是什么？", "neural network neurons layers synapses key terms", "Deep Learning", "07-DeepLearning/README.md"),
    ("neural_network_forward_pass", "神经网络如何从输入层计算到输出层？", "neural network input hidden output layers forward procedure", "Deep Learning", "07-DeepLearning/README.md"),
    ("neural_network_learning", "神经网络训练时主要需要学习什么参数？", "neural network learn weights synapses", "Deep Learning", "07-DeepLearning/README.md"),

    # 08-DimensionalityReduction/01-PrincipalComponentAnalysis.md
    ("pca", "PCA 是什么？", "principal component analysis PCA dimensionality reduction", "Dimensionality Reduction", "08-DimensionalityReduction/01-PrincipalComponentAnalysis.md"),
    ("pca_steps", "PCA 如何通过矩阵与向量运算改变输入方向？", "PCA functioning matrix vector multiplication rotates input vectors", "Dimensionality Reduction", "08-DimensionalityReduction/01-PrincipalComponentAnalysis.md"),
    ("pca_advantages", "PCA 降维有哪些优点？", "principal component analysis advantages dimensionality reduction", "Dimensionality Reduction", "08-DimensionalityReduction/01-PrincipalComponentAnalysis.md"),
    ("pca_feature_extraction", "PCA 如何从原始维度生成解释方差的新变量？", "PCA original dimensions produces new variables explain variance", "Dimensionality Reduction", "08-DimensionalityReduction/01-PrincipalComponentAnalysis.md"),

    # 08-DimensionalityReduction/README.md
    ("dimensionality_reduction_overview", "降维有哪两种基本思路？", "dimensionality reduction feature selection feature extraction", "Dimensionality Reduction", "08-DimensionalityReduction/README.md"),
    ("feature_selection_vs_extraction", "特征选择和特征提取有什么区别？", "feature selection discard features feature extraction create new features", "Dimensionality Reduction", "08-DimensionalityReduction/README.md"),
    ("dimensionality_methods", "Backward Elimination、PCA 和 LDA 分别属于哪类降维方法？", "dimensionality reduction backward elimination PCA LDA methods", "Dimensionality Reduction", "08-DimensionalityReduction/README.md"),

    # 11-TimeSeries/01-Introduction.md
    ("time_series_overview", "时间序列有哪些类型和特征？", "time series types features introduction", "Time Series", "11-TimeSeries/01-Introduction.md"),
    ("time_series_autocorrelation", "时间序列中的自相关是什么？", "time series autocorrelation covariance correlation lags", "Time Series", "11-TimeSeries/01-Introduction.md"),
    ("detect_autocorrelation", "可以用哪些检验检测残差自相关？", "detect autocorrelation Durbin Watson runs test chi-squared residuals", "Time Series", "11-TimeSeries/01-Introduction.md"),
    ("time_series_seasonality", "如何用平滑、过滤或 STL 处理季节性？", "time series seasonality smoothing filtering STL moving averages", "Time Series", "11-TimeSeries/01-Introduction.md"),
    ("time_series_forecasting", "时间序列预测需要关注哪些组成和方法？", "time series forecasting trend seasonality autocorrelation", "Time Series", "11-TimeSeries/01-Introduction.md"),

    # 11-TimeSeries/01-TimeSeriesInR.md
    ("time_series_r", "R 如何处理时间序列？", "time series in R xts operations", "Time Series", "11-TimeSeries/01-TimeSeriesInR.md"),
    ("xts_subsetting", "如何按日期区间截取 XTS 时间序列？", "R XTS subsetting date range time series", "Time Series", "11-TimeSeries/01-TimeSeriesInR.md"),
    ("xts_merge_na", "合并多个 XTS 对象时如何处理 NA？", "merge XTS objects NA handling R", "Time Series", "11-TimeSeries/01-TimeSeriesInR.md"),
    ("xts_lag_lead", "R 时间序列如何做 lagging 和 leading？", "XTS lagging leading time series R", "Time Series", "11-TimeSeries/01-TimeSeriesInR.md"),

    # 11-TimeSeries/README.md
    ("time_series_chapter", "什么样的数据叫时间序列？", "time series data fixed interval stochastic process", "Time Series", "11-TimeSeries/README.md"),
    ("time_series_chapter_topics", "时间序列章节包含哪些主题？", "time series chapter key terms types features autocorrelation", "Time Series", "11-TimeSeries/README.md"),
    ("time_series_implementation_index", "时间序列章节在哪里介绍 R 实现？", "time series implementation in R chapter index", "Time Series", "11-TimeSeries/README.md"),

    # 12-ConstraintSatisfactionProblems/README.md
    ("csp_overview", "什么是约束满足问题？", "constraint satisfaction problems CSP definitions", "Constraint Satisfaction Problems", "12-ConstraintSatisfactionProblems/README.md"),
    ("csp_structure", "CSP 由变量、取值域和约束怎样组成？", "CSP structure variables domains constraints", "Constraint Satisfaction Problems", "12-ConstraintSatisfactionProblems/README.md"),
    ("csp_solving", "约束满足问题通常如何求解？", "constraint satisfaction problem solving mechanism search assignment", "Constraint Satisfaction Problems", "12-ConstraintSatisfactionProblems/README.md"),

    # 13-Appendix/01-Programming/01-R/01-DplyrTutorial.md
    ("dplyr", "dplyr 怎么操作数据？", "dplyr manipulate data select mutate filter arrange", "Appendix", "13-Appendix/01-Programming/01-R/01-DplyrTutorial.md"),
    ("dplyr_group_summarise", "dplyr 如何分组并汇总数据？", "dplyr group_by summarise helper functions", "Appendix", "13-Appendix/01-Programming/01-R/01-DplyrTutorial.md"),
    ("dplyr_filter_arrange", "dplyr 如何筛选并排序行？", "dplyr filter arrange rows", "Appendix", "13-Appendix/01-Programming/01-R/01-DplyrTutorial.md"),
    ("dplyr_left_join", "dplyr 的 left join 如何使用键连接数据？", "dplyr left_join keys joining data", "Appendix", "13-Appendix/01-Programming/01-R/01-DplyrTutorial.md"),

    # 13-Appendix/01-Programming/02-Python/01-Numpy.md
    ("numpy", "NumPy 数组怎么用？", "NumPy arrays Python tutorial", "Appendix", "13-Appendix/01-Programming/02-Python/01-Numpy.md"),
    ("numpy_array_vs_list", "NumPy array 和 Python list 有什么区别？", "NumPy array versus Python list FAQ", "Appendix", "13-Appendix/01-Programming/02-Python/01-Numpy.md"),
    ("numpy_dot_outer_products", "NumPy 如何计算点积和外积？", "NumPy dot product outer product vectors", "Appendix", "13-Appendix/01-Programming/02-Python/01-Numpy.md"),
    ("numpy_linear_system", "NumPy 如何求解线性方程组？", "NumPy solve linear system linear algebra matrix", "Appendix", "13-Appendix/01-Programming/02-Python/01-Numpy.md"),

    # 13-Appendix/01-Programming/02-Python/02-MatPlotLib.md
    ("matplotlib", "Matplotlib 怎么画图？", "Matplotlib Python plotting", "Appendix", "13-Appendix/01-Programming/02-Python/02-MatPlotLib.md"),
    ("matplotlib_plot_types", "Matplotlib 如何画折线图、散点图、直方图和箱线图？", "Matplotlib line scatter histogram box plots", "Appendix", "13-Appendix/01-Programming/02-Python/02-MatPlotLib.md"),
    ("matplotlib_export", "Matplotlib 如何定制并导出图像？", "Matplotlib customize export save plots", "Appendix", "13-Appendix/01-Programming/02-Python/02-MatPlotLib.md"),

    # 13-Appendix/01-Programming/02-Python/03-Pandas.md
    ("pandas", "Pandas DataFrame 怎么用？", "Pandas Python dataframes tutorial", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),
    ("pandas_create_dataframe", "如何从字典或列表创建 Pandas DataFrame？", "Pandas create dataframe dictionaries lists", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),
    ("pandas_read_data", "Pandas 如何读取 CSV、Excel、SQL 和 HDF5？", "Pandas read import CSV Excel SQL HDF5", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),
    ("pandas_loc_iloc", "Pandas 的 loc 和 iloc 如何选择数据？", "Pandas loc iloc selecting indexing rows columns", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),
    ("pandas_time_series", "Pandas 如何计算时间序列移动平均？", "Pandas time series moving averages", "Appendix", "13-Appendix/01-Programming/02-Python/03-Pandas.md"),

    # 13-Appendix/01-Programming/02-Python/04-SciPy.md
    ("scipy", "SciPy 怎么读取 Matlab 文件？", "SciPy read import Matlab files loadmat", "Appendix", "13-Appendix/01-Programming/02-Python/04-SciPy.md"),
    ("scipy_loadmat", "scipy.io.loadmat() 返回什么对象？", "scipy io loadmat returns dictionary object", "Appendix", "13-Appendix/01-Programming/02-Python/04-SciPy.md"),
    ("scipy_matlab_objects", "为什么一个 Matlab 文件可能导入多个对象？", "SciPy Matlab stored workspace multiple objects import", "Appendix", "13-Appendix/01-Programming/02-Python/04-SciPy.md"),

    # 13-Appendix/01-Programming/02-Python/05-urllib.md
    ("urllib", "Python 如何从网页读取数据？", "Python web libraries urllib requests import data", "Appendix", "13-Appendix/01-Programming/02-Python/05-urllib.md"),
    ("python_web_csv_json", "Python 如何从 URL 读取 CSV 和 JSON？", "Python web libraries read CSV JSON URL", "Appendix", "13-Appendix/01-Programming/02-Python/05-urllib.md"),
    ("python_web_scraping", "如何用 requests 和 Beautiful Soup 抓取网页？", "Python web scraping requests Beautiful Soup", "Appendix", "13-Appendix/01-Programming/02-Python/05-urllib.md"),

    # 13-Appendix/01-Programming/02-Python/README.md
    ("appendix_python", "Python 编程附录介绍了什么？", "Python programming appendix libraries data types functions", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),
    ("python_data_types", "Python tuple、dictionary 和 string 如何使用？", "Python data types tuples dictionaries strings", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),
    ("python_comprehensions", "Python 列表推导式和字典推导式怎么写？", "Python list comprehensions dictionary comprehensions conditional", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),
    ("python_functions_scope", "Python 函数的可变参数、作用域和 lambda 怎么用？", "Python functions variable arguments scope lambda", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),
    ("python_generators_exceptions", "Python 生成器和异常处理如何使用？", "Python generators exception handling custom errors", "Appendix", "13-Appendix/01-Programming/02-Python/README.md"),

    # 13-Appendix/02-ApplicationAreas/01-Introduction.md
    ("application_intro", "机器学习在金融服务中有哪些应用？", "machine learning applications financial services", "Appendix", "13-Appendix/02-ApplicationAreas/01-Introduction.md"),
    ("financial_data_sources", "金融机器学习使用哪些传统和新型数据源？", "financial machine learning traditional new data sources", "Appendix", "13-Appendix/02-ApplicationAreas/01-Introduction.md"),
    ("financial_ml_decisions", "金融数据需要支持哪些业务决策？", "decisions financial data machine learning", "Appendix", "13-Appendix/02-ApplicationAreas/01-Introduction.md"),
    ("financial_ml_risks", "金融机器学习有哪些潜在问题和风险？", "financial machine learning potential problems application areas", "Appendix", "13-Appendix/02-ApplicationAreas/01-Introduction.md"),

    # README.md
    ("ml_notes_overview", "这份机器学习笔记覆盖哪些主题？", "machine learning notes topics index regression classification clustering", "Root", "README.md"),
    ("ml_notes_navigation", "从总目录如何找到回归、分类和聚类章节？", "machine learning book index regression classification clustering chapters", "Root", "README.md"),
    ("ml_notes_appendix_index", "机器学习笔记的先修知识和附录入口在哪里？", "machine learning notes prerequisites appendix index", "Root", "README.md"),
)


# These labels point to missing or too-short files in the current corpus. They
# remain visible for data-quality tracking but never enter primary metrics.
DATA_QUALITY_CASE_ROWS = (
    ("decision_tree", "决策树是什么？", "decision tree classification algorithm", "Classification", "02-Classification/05-DecisionTree.md"),
    ("random_forest", "随机森林是什么？", "random forest classification ensemble decision trees", "Classification", "02-Classification/06-RandomForest.md"),
    ("gaussian_mixture", "高斯混合模型是什么？", "gaussian mixture model clustering expectation maximization", "Clustering", "03-Clustering/03-GaussianMixtureModels.md"),
    ("time_series_state_space", "状态空间模型是什么？", "state space models time series", "Time Series", "11-TimeSeries/StateSpaceModels.md"),
    ("numpy_matrix", "numpy matrix 怎么用？", "numpy matrix tutorial Python", "Prerequisites", "00-Prerequisites/numpyMatrixTutorial.md"),
    ("logistic_regression", "逻辑回归是什么？", "logistic regression classification", "Classification", "02-Classification/01-LogisticRegression.md"),
    ("knn", "KNN 分类如何工作？", "k nearest neighbors KNN classification", "Classification", "02-Classification/02-knn.md"),
    ("svm", "支持向量机是什么？", "support vector machines SVM classification", "Classification", "02-Classification/03-SupportVectorMachines.md"),
    ("hidden_markov", "隐马尔可夫模型是什么？", "hidden markov models HMM", "Classification", "02-Classification/07-HiddenMarkovModels.md"),
    ("reinforcement_overview", "强化学习是什么？", "reinforcement learning", "Reinforcement Learning", "05-ReinforcementLearning/README.md"),
    ("recommendation_overview", "推荐系统是什么？", "recommendation engines", "Recommendation Engines", "09-RecommendationEngines/README.md"),
    ("boosting_overview", "Boosting 是什么？", "model selection boosting", "Model Selection and Boosting", "10-ModelSelectionAndBoosting/README.md"),
    ("computer_science", "计算机科学基础", "computer science prerequisites", "Prerequisites", "00-Prerequisites/ComputerScience.md"),
    ("appendix_r", "R 编程附录", "R programming appendix", "Appendix", "13-Appendix/01-Programming/01-R/README.md"),
    ("programming_overview", "编程附录", "programming appendix", "Appendix", "13-Appendix/01-Programming/README.md"),
)
