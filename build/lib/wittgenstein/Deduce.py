import numpy as np
import pandas as pd
from sympy import *
from sympy.logic.boolalg import to_cnf
from sympy.logic.boolalg import to_dnf
import copy
from sklearn.feature_selection import mutual_info_classif
import operator
import heapq

#variables:
#name of the node, a count
#nodelink used to link similar items
#parent vaiable used to refer to the parent of the node in the tree
#node contains an empty dictionary for the children in the node
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {} 

#increments the count variable with a given amount    
    def inc(self, numOccur):
        self.count += numOccur
#display tree in text. Useful for debugging        
    def disp(self, ind=1):
        print ('  |'*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
        
def createTree(dataSet, minSup=1, target_Var=[None],target_Var_Pos=None): #create FP-tree from dataset but don't mine
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable):  #remove items not meeting minSup
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        tmp=[]
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            if target_Var_Pos==True:#target_Var 擺第一
                for i in target_Var:
                    if localD.get(i,None)!=None:
                        tmp.append(i)
                        localD.pop(i,None)
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                tmp.extend(orderedItems)
                orderedItems=tmp
            elif target_Var_Pos==False: #target_Var 擺最後
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                for i in target_Var:
                    if localD.get(i,None)!=None:
                        orderedItems.remove(i)
                        orderedItems.append(i)
            else:
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]               
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    headerTable['leaf']=headerTable.get('leaf',[None,None])
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode      

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = retDict.get(frozenset(trans), 0) + 1
    return retDict

def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

# =============================================================================
class RIPPER_Deduce:
    def __init__(self,df):
        self.columns_len = len(df.columns) 
        self.df_cols = list(df.columns)
        self.col_dict = {}
        self.col_explain = {}
        
    def Infomation_Gain(self,df,target_class):
        X = df.iloc[:,self.columns_len:len(df)]
        Y = df[target_class]
        info_gain_value = mutual_info_classif(X ,Y ,discrete_features=True)
        info_gain_key =  list(df.iloc[:,self.columns_len:len(df)].columns)
        dictionary = dict(zip(info_gain_key, info_gain_value))
        return dictionary
    
    def feature_select(self,df,df_test,target_class,n,round_n=4):
        if len(df_test.columns) == self.columns_len:
            return df,0
        dict_tmp = self.Infomation_Gain(df_test,target_class=target_class)
        topitems = heapq.nlargest(len(dict_tmp), dict_tmp, key=dict_tmp.get)
        #print(topitems)
        df_list = []
        for col_i in range(len(topitems)):
            if len(df_list) >= n:
                break
            if col_i == 0:
                df_list.append(topitems[col_i])
            elif round(dict_tmp[topitems[col_i]],round_n) < round(dict_tmp[topitems[col_i-1]],round_n):
                #print(dict_tmp[topitems[col_i-1]],topitems[col_i-1])
                #print(dict_tmp[topitems[col_i]],topitems[col_i])
                df_list.append(topitems[col_i])
        print('---- Top %s feature score [0-1] ----' %n)
        for col in df_list[:n]:
            print('  %s : %s'%(col,dict_tmp[col])) 
        print('------------------------------------')
        col_list =[]
        for col in df.columns:
            if col in df_list:
                col_list.append(col)
        return  pd.concat([df[self.df_cols],df[col_list]],axis=1),dict_tmp[topitems[0]]
        
    def parser(self,df, coder_list, symbol):
        #print('----- %sparser -----'%symbol)
        #print(coder_list)
        if symbol == '&':
            df_dual_encoding = np.zeros(len(df),'int8') ==0
            for flag in coder_list:
                flag_name,code_value = self.feature_parser(flag)
                #print('dtype:%s' %df[flag_name].dtype)
                if df[flag_name].dtype == 'bool':
                    #print('parser:_bool_')
                    df_dual_encoding = (df_dual_encoding) & (df[flag_name] == code_value[0])
                elif df[flag_name].dtype == 'object':
                    #print('parser:_object_')
                    df_dual_encoding = (df_dual_encoding) & (df[flag_name] == str(code_value[0]))
                else:
                    #print('parser:_NUM_')
                    if len(code_value)>1:
                        df_dual_encoding = (df_dual_encoding) & ((df[flag_name] >= float(code_value[0])) & (df[flag_name] <= float(code_value[1])))
                    else:
                        df_dual_encoding = (df_dual_encoding) & ((df[flag_name] >= float(code_value[0])) & (df[flag_name] <= float(code_value[0])))
        elif symbol == '|':
            df_dual_encoding = np.zeros(len(df),'int8') >0
            for flag in coder_list:
                flag_name,code_value = self.feature_parser(flag)
                #print('dtype:%s' %df[flag_name].dtype)
                if df[flag_name].dtype == 'bool':
                    #print('parser:_bool_')
                    df_dual_encoding = (df_dual_encoding) | (df[flag_name] == code_value[0])
                elif df[flag_name].dtype == 'object':
                    #print('parser:_object_')
                    df_dual_encoding = (df_dual_encoding) | (df[flag_name] == str(code_value[0]))
                else:
                    #print('parser:_NUM_')
                    if len(code_value)>1:
                        df_dual_encoding = (df_dual_encoding) | ((df[flag_name] >= float(code_value[0])) & (df[flag_name] <= float(code_value[1])))
                    else:
                        df_dual_encoding = (df_dual_encoding) | ((df[flag_name] >= float(code_value[0])) & (df[flag_name] <= float(code_value[0])))

        return df_dual_encoding
    def feature_parser(self,flag):
        #print(flag)
        if flag[0] == '~':
            flag_name = flag[2:-1]
            flag_name = flag_name.replace('_IS__NEG_','=-')
            flag_name = flag_name.replace('_IS_','=')
            flag_name = flag_name.replace('_TO__NEG_','--')
            flag_name = flag_name.replace('_TO_','-')
            #print('feature_parser:_FALSE_')
            return flag_name,[False]
        elif len(flag.rsplit('_IS_',1)) == 1 or  flag[-1] == ')':
            flag_name = flag
            flag_name = flag_name.replace('_IS__NEG_','=-')
            flag_name = flag_name.replace('_IS_','=')
            flag_name = flag_name.replace('_TO__NEG_','--')
            flag_name = flag_name.replace('_TO_','-')
            #print('feature_parser:_TRUE_')
            return flag_name,[True]
        elif len(flag.rsplit('_IS_',1)[1].rsplit('_TO_',1)) >1:
            flag = flag.replace('_NEG_','-')
            flag_name = flag.rsplit('_IS_',1)[0]
            code_value = np.array(flag.rsplit('_IS_',1)[1].rsplit('_TO_',1)).astype('float')
            #print('feature_parser:_NUM_')
            return flag_name,code_value
        elif len(flag.rsplit('_IS_',1)[1].rsplit('_TO_',1)) ==1:
            flag_name = flag.rsplit('_IS_',1)[0]
            code_value = flag.rsplit('_IS_',1)[1].rsplit('_TO_',1)
            #print('feature_parser:_object_')
            return flag_name,code_value
        else:
            print('%s\tERROR!!'%symbol)
    
    def duality_encoder(self,df, coder_list=[], symbol='&',nor_form = 'normal'):
        if len(coder_list) ==0: return ''
        coder_list_tmp = copy.deepcopy(coder_list)
        for i in range(len(coder_list)):
            coder_list_tmp[i] = coder_list_tmp[i].replace('-','_TO_')
            coder_list_tmp[i] = coder_list_tmp[i].replace('=_TO_','_IS__NEG_')
            coder_list_tmp[i] = coder_list_tmp[i].replace('=','_IS_')
            coder_list_tmp[i] = coder_list_tmp[i].replace('_TO__TO_','_TO__NEG_')
            
        logic_sentence = ''
        if len(coder_list_tmp) == 1:
            logic_sentence = '%s'
            logic_sentence = logic_sentence %coder_list_tmp[0]
        else:
            for i in range(len(coder_list_tmp)):
                logic_sentence = logic_sentence + '(%s) '
            logic_sentence = logic_sentence[:-1].replace(' ', str(symbol))
            logic_sentence = logic_sentence %tuple(coder_list_tmp)
        coder_normal = logic_sentence.replace('_IS__NEG_','=-')
        coder_normal = coder_normal.replace('_IS_','=')
        coder_normal = coder_normal.replace('_TO__NEG_','--')
        coder_normal = coder_normal.replace('_TO_','-')
        if nor_form == 'normal':
            coder_name = coder_normal
        elif nor_form == 'short':
            if len(df.columns) == self.columns_len:
                coder_short = 'var_0'
            else:
                coder_short = 'var_%s' %(int(list(df.columns)[-1].split('_')[1])+1)
            coder_name = coder_short
            self.col_explain[coder_short] = self.col_explain.get(coder_short,coder_normal)
        
        self.col_dict[coder_normal] = self.col_dict.get(coder_normal,False)
        if self.col_dict[coder_normal] == False:
            encoding = self.parser(df, coder_list_tmp, symbol)
            df[coder_name] = encoding
            self.col_dict[coder_normal] = True
            #print('coder_name:%s coder_normal:%s'%(coder_name,coder_normal))
        return coder_name

    def rule_Deduce(self,df, tree_node, absorpt=[],nor_form = 'normal'):
        # 吸收
        coder_name = self.duality_encoder(df, absorpt, symbol='&',nor_form = nor_form)
        #print(tree_node.name,coder_name)
        
        if tree_node.name != 'Null Set':
            absorpt.append(tree_node.name)
        coder_list = []
        for child in tree_node.children:
            coder_mem = self.rule_Deduce(df, tree_node.children[child], absorpt, nor_form = nor_form)
            coder_list.append(coder_mem)
            absorpt.pop()
        if tree_node.name == 'Null Set':
            return
        
        # 內構
        if (len(coder_list)>1) & (tree_node.name != 'Null Set'):
            coder_name = self.duality_encoder(df, coder_list, symbol='|',nor_form = nor_form)
            intra_list = [coder_name]
        else:
            intra_list = coder_list
            
        # 辨識
        identification_list=[]
        identification_list.extend(intra_list)
        if tree_node.parent.name != 'Null Set':
            identification_list.append(tree_node.name)
            coder_name = self.duality_encoder(df, identification_list, symbol='&',nor_form = nor_form)
        return coder_name
    
    def ruleset_parser(self,ruleset_):
        ruleset = str(ruleset_).replace('[','')
        ruleset = ruleset.replace(']','')
        ruleset = ruleset.split(' V ')
        ruleset_list = []
        for i in range(len(ruleset)):
            rule_str = self.rule_conjunct_parser(ruleset[i])
            ruleset_list.append(rule_str)
        return np.array(ruleset_list)

    def rule_conjunct_parser(self,rule):
        rule_conjunct = []
        rule = rule + '^'
        stack = 0
        rule_tmp = ''
        for mem in list(rule):
            #print(stack,'\t',mem,'\t',rule_tmp)
            if mem == '(':
                stack += 1
            elif mem == ')':
                stack -= 1
            if mem != '^':
                rule_tmp = rule_tmp + mem
            else:
                if stack != 0:
                    rule_tmp = rule_tmp + mem
                else:
                    #print(rule_tmp)
                    rule_name = rule_tmp.rsplit('=',1)[0]
                    rule_value = rule_tmp.rsplit('=',1)[1].lower()
                    if rule_value == 'true':
                        rule_conjunct.append('%s' %rule_name)
                    elif rule_value == 'false':
                        rule_conjunct.append('~(%s)' %rule_name)
                    else:
                        rule_conjunct.append(rule_tmp)
                    rule_tmp = ''
        return rule_conjunct
    def ruleset_tree(self,rule_set):
        #建立 FP_TREE
        initSet = createInitSet(rule_set)
        rstree,rtree_HeaderTab=createTree(initSet,0)
        return rstree

# =============================================================================
if __name__ == "__main__":
    tmp = [['aa',True,True,False,True,True,True,True,-5],
           ['aa',False,True,True,False,True,True,True,-4],
           ['aa',False,False,True,True,False,True,True,-3],
           ['aa',True,True,True,True,True,False,True,-2],
           ['aa',False,False,True,True,True,True,False,-1],
           ['aa',True,True,False,True,True,True,False,0],
           ['aa',False,True,True,True,True,True,False,6],
           ['bb',True,True,True,False,True,True,False,7],
           ['bb',True,False,False,True,True,False,False,8],
           ['bb',False,False,True,True,False,True,True,9],
           ['bb',True,True,True,True,True,True,False,10],
           ['bb',True,True,False,True,False,True,False,11],
           ['bb',True,False,True,True,True,True,True,12],
           ['bb',False,True,False,False,True,True,False,13],
           ['bb',True,True,True,False,False,True,False,14],
           ['bb',True,False,True,True,False,True,True,15]]
    df = pd.DataFrame(tmp,columns = ['a','b','c','d','e','f','g','h','i'])

    rule_list = '[a=aa^b=False^d=True] V [a=aa^b=False^e=True^g=True] V [a=aa^b=False^e=True^h=False^i=-5--1] V [a=aa^b=False^f=True] V [a=aa^c=True]'

    
    RIPPER_rule_Deduce = RIPPER_Deduce(df)
    np_rlist = RIPPER_rule_Deduce.ruleset_parser(rule_list)
    rstree =RIPPER_rule_Deduce.ruleset_tree(np_rlist)
    RIPPER_rule_Deduce.rule_Deduce(df,rstree,nor_form = 'short')
    
    RIPPER_rule_Deduce.col_explain['var_14']
    RIPPER_rule_Deduce.col_explain
    #挖掘 FP_TREE
   #findPrefixPath('i', myHeaderTab['i'][1])
    #myFPtree.children['a']
# =============================================================================
# else:
#     print('->This is FP_tree Data Structure Package, Function list and Data Format is:')
#     print('1.<Data Set> format is -- np.array() --')
#     print('2.createInitSet(<Data Set>):Input <Data Set>, Output FPtree,myHeaderTab')
#     print('3.findPrefixPath(find_Vari,myHeaderTab[Vari][1]):Get about Vari information in FPtree')
#     print('4.treeNode.disp():Show FPtree Structure')
#     
# =============================================================================
