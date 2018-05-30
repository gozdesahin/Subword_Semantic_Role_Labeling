#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Read Finnish SRL file in a structured way
Used by Loader during training and testing

"""
import codecs

PRED_SENSE_COL = 9

class conllud_fin:
    def __init__(self, fpath=None):
        self.sents = []
        self.totPredCnt = 0
        self.totArgCnt = 0
        self.totTokCnt = 0
        self.firstsense = 0
        if fpath is not None:
            self.sents= self.read_file(fpath)
        else:
            print("File can not be opened, check path")

    def get_labels_for_pred(self, csent, predid):
        labels = ["_"]*len(csent.tokens)
        strToBeFound = str(predid)+":PB"
        for i,tok in enumerate(csent.tokens):
            if strToBeFound in tok.argInfo:
                argParts = tok.argInfo.split("|")
                for part in argParts:
                    if part.startswith(strToBeFound):
                        lab = part[len(strToBeFound):]
                        labels[i] = lab
                        csent.argcnt+=1
        return labels

    def read_file(self, file_path):
        fin = codecs.open(file_path, encoding='utf-8')
        strIn = fin.read()
        fin.close()
        conllsentences = []
        # get sentences
        sentences = strIn.split("\n\n")
        for sent in sentences:
            if(len(sent)>0):
                lines = sent.split("\n")
                rows = []
                predid = 0
                csent = conllUDsent()
                # make a tab separated rows list
                for line in lines:
                    if line.startswith(u"#"):
                        continue
                    # make a new token
                    ctoken = conllUDtoken(line.split("\t"))
                    csent.add_token(ctoken)
                    if ctoken.ispred:
                        predid = predid+1
                        ctoken.set_predid(predid)
                    rows.append(line.split())
                csent.rows = rows
                for pi in csent.predind:
                    l = self.get_labels_for_pred(csent, pi+1)
                    csent.labels.append(l)
                csent.tokenWords = [tok.word for tok in csent.tokens]
                csent.tokenLemmas = [tok.lemma for tok in csent.tokens]
                csent.tokenOracles = [tok.oracle for tok in csent.tokens]

                self.totTokCnt += len(csent.tokens)
                self.totArgCnt += csent.argcnt
                self.totPredCnt += csent.predcnt
                conllsentences.append(csent)

        print "%d sentences %d tokens %d predicates and %d arguments are successfully read" %(len(conllsentences),self.totTokCnt,self.totPredCnt, self.totArgCnt)
        print "%d number of predicates with first sense" % (self.firstsense)
        return conllsentences


class conllUDsent:
    def __init__(self):
        self.tokens = []
        # register predicate words
        self.predWords = []
        # register predicate lemmas
        self.predLemmas = []
        # save predicate indices (for position features)
        # starts from zero
        self.predind = []
        # whole column of semantic labels for each predicate
        self.labels = []
        # token words and lemmas
        self.tokenWords = []
        self.tokenLemmas = []
        self.predcnt = 0
        self.argcnt = 0

    def add_token(self, token):
        self.tokens.append(token)
        if(token.ispred):
            self.predWords.append(token.word)
            self.predLemmas.append(token.lemma)
            self.predind.append(token.id-1)
            self.predcnt+=1

class conllUDtoken:
    def __init__(self, fields):
        self.id = int(fields[0])
        self.word = fields[1]
        self.lemma = fields[2]
        self.pos = fields[3]
        self.ppos = fields[4]
        self.feat = fields[5]
        self.head = fields[6]
        self.deplab = fields[7]
        self.deplabel = fields[7]
        self.predid = -1
        self.predsense = u"_"
        self.ispred = ("PBSENSE=" in fields[9])
        self.argInfo = fields[8]
        if self.ispred:
            fieldParts = fields[9].split("|")
            for part in fieldParts:
                if part.startswith("PBSENSE="):
                    sense = part[8:]
                    self.predsense = sense
        # the order of the predicate (if it is a predicate)
        self.oracle = ""
        infmorph = "+"+self.pos+"+"+("+".join(self.feat.split("|")))
        self.oracle = "word:"+self.word+"+lemma:"+ self.lemma+infmorph

    def set_predid(self, id):
        self.predid=id
