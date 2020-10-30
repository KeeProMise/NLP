


def set_patten(patten):
    patten_dict = {}
    for start ,end in patten:
        patten_dict[start] = end
    return patten_dict

def ishavetag(patten,tag):
    for start,end in patten:
        if tag==start or tag==end:
            return True
    return False

def entity_number(patten_dict,parts):
    num=0
    for part in parts:
        i=0
        for i in range(len(part)) :
            word = part[i]
            if word in patten_dict:
                num+=1
    return num


def match_number(patten_dict,y_true,y_predict):
    num = 0
    for i in range(len(y_true)):
        y_true_part = y_true[i]
        y_predict_part = y_predict[i]
        j=0
        while j < len(y_true_part):
            if y_true_part[j] in patten_dict and y_true_part[j]==y_predict_part[j]:
                true = 1
                z = j+1
                while z < len(y_true_part) and y_true_part[z]==patten_dict[y_true_part[j]]:
                    if y_true_part[z]!=y_predict_part[z]:
                        true=0
                        break
                    z+=1
                j=z
                if true==1:num+=1
                continue
            j+=1
    return num

def NER_precision(y_predict_entity,true_entity):
    return true_entity/y_predict_entity

def NER_recall(y_true_entity,true_entity):
    return true_entity/y_true_entity

def NER_f1_score(precision_score,recall_score):
    return (2*recall_score*precision_score)/(precision_score+recall_score)

def NER_score(patten,y_true,y_predict):
    patten_dict = set_patten(patten)
    y_predict_entity = entity_number(patten_dict,y_predict)
    print("模型预测的实体数量：",y_predict_entity)
    y_true_entuty = entity_number(patten_dict,y_true)
    print("真实的实体数量：",y_true_entuty)
    true_entity = match_number(patten_dict,y_true,y_predict)
    print("正确匹配的实体数量：",true_entity)
    precision = NER_precision(y_predict_entity,true_entity)
    print("precision=",precision)
    recall = NER_recall(y_true_entuty,true_entity)
    print("recall=",recall)
    f1_score = NER_f1_score(precision,recall)
    print("f1_score=",f1_score)
    return (precision,recall,f1_score)

#例：
# list = [[1,1,1,0,0,1,0,2,2,0,1,2,1,2,2,2,2,2,0,1],
#         [1],[0,0,0,0,0,0,0,2,2,0,1,2],[0,0,0,0,0]]
#
# list2 = [[1,1,1,0,0,2,0,2,2,0,1,2,1,2,2,2,2,2,0,1,""],
#         [1],[0,0,0,0,0,0,0,2,2,0,1,2],[0,0,0,0,0]]
#
# patten = [(1,2)]
# NER_score(patten,list,list2)
#
# print(entity_number({1:2},list))
# print(match_number({1:2},list,list2))





