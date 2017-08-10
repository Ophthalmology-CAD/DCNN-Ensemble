from jiang_classification_test_256 import *
from jiang_classification_test_128 import *
from jiang_classification_test_alexnet import *
from pre_net import *
import os

fenlei = 'shenqian-6'
model = ['6','2','4','8','1']
recurrence_file='qian'
surgery_file='shen'
image='5'

for i in range(0,1):
    print i
    net_alexnet,net_googlenet,net_resnet_50=pre_net(fenlei,model[i])
    ###alexnet

    other_alexnet,surgery_alexnet=jiang_classification_test_alexnet(net_alexnet,recurrence_file,surgery_file,image,model[i],fenlei)


    ###googlenet

    other_googlenet,surgery_googlenet=jiang_classification_test_256(net_googlenet,recurrence_file,surgery_file,image,model[i],fenlei)

    ###resnet_50

    other_resnet_50,surgery_resnet_50=jiang_classification_test_128(net_resnet_50,recurrence_file,surgery_file,image,model[i],fenlei)


    #test_other=caffe_root+'myself/slitlamp-twoclass-1-10data-256/'+recurrence_file+'/'+image
    #test_surgery=caffe_root+'myself/slitlamp-twoclass-1-10data-256/'+surgery_file+'/'+image
    test_other='/home/shiyan/caffe-five-cost-sensitive/'+'myself/external_test_data/web-256/'+recurrence_file
    test_surgery='/home/shiyan/caffe-five-cost-sensitive/'+'myself/external_test_data/web-256/'+surgery_file

    sum_surgery = 0
    error_surgery_number = 0
    list_surgery0 = []
    list_surgery1 = []
    list_surgery2 = []
    # surgeryp=[]
    print '111111'
    for root, dirs, files in os.walk(test_surgery):
        index = 0
        for file in files:
            print file
            surgery_treat = []
            print surgery_alexnet[index]
            print surgery_googlenet[index]
            print surgery_resnet_50[index]
            surgery_treat.append(float(surgery_alexnet[index][0] + surgery_googlenet[index][0] + surgery_resnet_50[index][0]) / 3)
            surgery_treat.append(float(surgery_alexnet[index][1] + surgery_googlenet[index][1] + surgery_resnet_50[index][1]) / 3)
            index_surgery = surgery_treat.index(max(surgery_treat))
            print surgery_treat
            print 'predicted max:', max(surgery_treat)
            print 'predicted class:', surgery_treat.index(max(surgery_treat))
            print '111111'
            sum_surgery = sum_surgery + 1
            # surgeryp.append(prediction)
            if index_surgery == 0:
                error_surgery_number = error_surgery_number + 1
                list_surgery0.append(file + '\t' + str(surgery_alexnet[index]) + str(surgery_googlenet[index]) + str(surgery_resnet_50[index]) + '\t' +  'class:' + str(index_surgery) + '\t' + 'average:' + str(surgery_treat))
                #list_surgery0.append(file + '\t' + 'predicted class:' + str(index_surgery) + '\t' + 'predicted:' + str(surgery_treat))
            if index_surgery == 1:
                list_surgery1.append(file + '\t' +str(surgery_alexnet[index]) + str(surgery_googlenet[index]) + str(surgery_resnet_50[index]) + '\t' + 'class:' + str(index_surgery) + '\t' + 'average:' + str(surgery_treat))
                #list_surgery1.append(file + '\t' + 'predicted class:' + str(index_surgery) + '\t' + 'predicted:' + str(surgery_treat))
            index = index +1
    list_surgery2.append('sum_surgery:' + str(sum_surgery) + '\t' + 'error_surgery_number:' + str(error_surgery_number) + '\t' + 'the surgery accuracy is:' + str(float((sum_surgery) - (error_surgery_number)) / float(sum_surgery)))

    sum_other = 0
    error_other_number = 0
    list_other0 = []
    list_other1 = []
    list_other2 = []
    for root, dirs, files in os.walk(test_other):
        index = 0
        for file in files:
            other_treat = []
            print other_alexnet[index]
            print other_googlenet[index]
            print other_resnet_50[index]
            other_treat.append(float(other_alexnet[index][0] + other_googlenet[index][0] + other_resnet_50[index][0]) / 3)
            other_treat.append(float(other_alexnet[index][1] + other_googlenet[index][1] + other_resnet_50[index][1]) / 3)
            index_other = other_treat.index(max(other_treat))
            print other_treat
            print 'predicted max:', max(other_treat)
            print 'predicted class:', other_treat.index(max(other_treat))
            print '111111'
            sum_other = sum_other + 1
            if index_other == 0:
                list_other0.append(file + '\t' + str(other_alexnet[index]) + str(other_googlenet[index]) + str(other_resnet_50[index]) + '\t' + 'class:' + str(index_other) + '\t' + 'average:' + str(other_treat))
            if index_other == 1:
                error_other_number = error_other_number + 1
                list_other1.append(file + '\t' + str(other_alexnet[index]) + str(other_googlenet[index]) + str(other_resnet_50[index]) + '\t' + 'class:' + str(index_other) + '\t' + 'average:' + str(other_treat))
            index = index + 1
    list_other2.append('sum_recurrence:' + str(sum_other) + '\t' + 'error_recurrence_number:' + str(error_other_number) + '\t' + 'the other accuracy is:' + str(float((sum_other) - (error_other_number)) / float(sum_other)))

    accuracy = float((sum_other + sum_surgery) - (error_other_number + error_surgery_number)) / float(sum_other + sum_surgery)
    print 'the accuracy is:', accuracy * 100
    list_surgery2.append('the total accuracy is:' + str(accuracy))


    RESULT_FILE = 'web_data_result/'+fenlei+'/average_train_'+model[i]+'.txt'


    file_object = open(RESULT_FILE, 'w')
    file_object.writelines('ERROR_RECURRENCE\n')
    for i in list_other0:
        file_object.writelines(i + '\n')
    for i in list_other1:
        file_object.writelines(i + '\n')
    for i in list_other2:
        file_object.writelines(i + '\n')
        file_object.writelines('-------------------------------------------------------\n')
        file_object.writelines('ERROR_SUGERY\n')
    for i in list_surgery0:
        file_object.writelines(i + '\n')
    for i in list_surgery1:
        file_object.writelines(i + '\n')
    for i in list_surgery2:
        file_object.writelines(i + '\n')
    file_object.close()





    


