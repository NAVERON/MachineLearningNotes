


from matplotlib import pyplot
import pandas
from sklearn import linear_model

def get_data(file_name): #从外界数据库读取数据，返回两个序列
    data = pandas.read_csv(file_name)  # 读取的格式注意，上面需要标题栏，后面根据标题来读
    x_parameter = [] #list not tuple
    y_parameter = []
    
    for sin_sqre,sin_price in zip(data["sqr"],data["pri"]):
        x_parameter.append([float(sin_sqre)])
        y_parameter.append([float(sin_price)])
    
    print("x = ",x_parameter, "\ny = ", y_parameter)
    return x_parameter, y_parameter


#训练模型
def linear_model_train(x_parameter, y_parameter, predict_value):
    regr = linear_model.LinearRegression()
    regr = regr.fit(x_parameter, y_parameter)
    #print(regr)
    predict_out = regr.predict(predict_value)
    predictions = {}
    predictions["intercept"] = regr.intercept_
    predictions["coefficient"] = regr.coef_
    predictions["predict_value"] = predict_out
    
    return predictions,regr

#显示结果图形
def show_linear(x_parameters, y_parameters, regr):
#     regr = linear_model.LinearRegression()
#     regr = regr.fit(x_parameters, y_parameters)
    pyplot.scatter(x_parameters, y_parameters)
    pyplot.plot(x_parameters, regr.predict(x_parameters),color='red',linewidth=2)
    pyplot.xticks(())
    pyplot.yticks(())
    pyplot.show()

if __name__ == "__main__":
    x,y = get_data("data.csv")
    predictvalue = 300
    result, regr = linear_model_train(x,y,predictvalue)
    
    print("intercept : ",result["intercept"])
    print("coefficient : ",result["coefficient"])
    print("prediction : ", result["predict_value"])
    
    show_linear(x, y, regr)










