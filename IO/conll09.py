#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Read CoNLL09 files in a structured way
Used by Loader during training and testing

Notes on languages:
German OOD: Should use predicted morphological tags
Spanish, Catalan: Long MWE are converted to abbreviations (e.g., Abraham_Lincoln -> AL)
Turkish: Derivational boundaries require special processing
"""

import codecs

FIXED_COL_CNT = 14

class conll09:
    def __init__(self, fpath=None, use_predicted=False):
        """
        Read file into a list of conll09sent objects
        :param fpath: path to CoNLL09 file
        :param use_predicted: Should be True if predicted morphological tags will be used,
                              otherwise gold tags will be used
        """
        self.sents = []
        self.totPredCnt = 0
        self.totArgCnt = 0
        self.totTokCnt = 0
        self.firstsense = 0
        self.use_predicted = use_predicted
        if fpath is not None:
            self.sents= self.read_file(fpath)
        else:
            print("File can not be opened, check path")

    def read_file(self, file_path):
        fin = codecs.open(file_path, encoding='utf-8')
        strIn = fin.read()
        fin.close()
        conllsentences = []
        sentences = strIn.split("\n\n")
        for sent in sentences:
            rows = []
            orgRows = []
            if(len(sent)>0):
                lines = sent.split("\n")
                predcnt = len(lines[0].split("\t"))-FIXED_COL_CNT
                self.totPredCnt += predcnt
                # create new conll09 sentence
                csent = conll09sent(predcnt)
                predid = 0
                tokenid = 0
                # for derivational boundaries
                prevToken = None
                firstInDeriv = True
                for line in lines:
                    orgRows.append(line.split())
                    # create a conll09 token
                    if firstInDeriv:
                        ctoken = conll09token(line.split("\t"), tokenid, self.use_predicted)
                        prevToken = ctoken
                        if(ctoken.ispred):
                            predid = predid+1
                            ctoken.set_predid(predid)
                            if(ctoken.predsense.endswith(".01")):
                                self.firstsense+=1
                        if(ctoken.word=="_"):
                            # update deriv value
                            firstInDeriv = False
                        csent.add_token(ctoken)
                        rows.append(line.split())
                        tokenid += 1
                    else:
                        fields = line.split("\t")
                        prevToken.update_token(fields)
                        rows[-1] = self.mergeLabels(rows[-1],fields)
                        # workaround for words with two predicates inside
                        if (fields[FIXED_COL_CNT-2] == 'Y'):
                            prevToken.ispred=True
                            csent.add_predicate(prevToken)
                        if fields[1]=="_":
                            firstInDeriv = False
                        else:
                            firstInDeriv = True
                # add token words
                csent.tokenWords = [tok.word for tok in csent.tokens]
                csent.tokenLemmas = [tok.lemma for tok in csent.tokens]
                csent.tokenOracles = [tok.oracle for tok in csent.tokens]

                self.totTokCnt += len(csent.tokens)
                # add labels for each predicate
                for i in range(predcnt):
                    l = [row[FIXED_COL_CNT+i] for row in rows]
                    csent.labels.append(l)
                    self.totArgCnt += (len(l)-l.count("_"))
                csent.orgRows = orgRows
                conllsentences.append(csent)
        print("%d sentences %d tokens %d predicates and %d arguments are successfully read" %(len(conllsentences),self.totTokCnt,self.totPredCnt, self.totArgCnt))
        print("%d number of predicates with first sense" % (self.firstsense))
        return conllsentences

    def mergeLabels(self, orgRow, fields):
        """
        Merge derivational boundaries in Turkish CoNLL09 data
        :param orgRow: original row
        :param fields: fields of the new row
        :return: newrow: merged row
        """
        newrow = orgRow
        for i in range(len(orgRow)):
            if (orgRow[i] == "_" and fields[i] != "_"):
                newrow[i] = fields[i]
        return newrow

class conll09sent:
    def __init__(self,predcnt):
        self.tokens = []
        # register predicate words
        self.predWords = []
        # register predicate lemmas
        self.predLemmas = []
        # save predicate indices (for position features)
        self.predind = []
        # whole column of semantic labels for each predicate
        self.labels = []
        # token words and lemmas
        self.tokenWords = []
        self.tokenLemmas = []
        self.predcnt = predcnt

    def add_token(self, token):
        self.tokens.append(token)
        if(token.ispred):
            self.predWords.append(token.word)
            self.predLemmas.append(token.lemma)
            self.predind.append(token.tokenid)

    def add_predicate(self, token):
        if(token.ispred):
            self.predWords.append(token.word)
            self.predLemmas.append(token.lemma)
            self.predind.append(token.tokenid)

class conll09token:
    def __init__(self, fields, tokenid, use_predicted=False):
        self.id = fields[0]
        self.ids = []
        self.ids.append(self.id)
        self.word = fields[1]
        self.lemma = fields[2]
        self.plemma = fields[3]
        # if the language is spanish or catalan, they merge mwe with "_" and
        # it causes memory problems for lstm
        if self.word != "_" and ("_" in self.word):
            str = ""
            for tpart in self.word.split("_"):
                if len(tpart)==0:
                    continue
                str+=tpart[0]
            self.word = str
            self.lemma = self.word
        self.pos = fields[4]
        self.ppos = fields[5]
        self.feat = fields[6]
        self.pfeat = fields[7]
        self.deplabel = fields[10]
        self.ispred = (fields[12]=='Y')
        # the order of the predicate (if it is a predicate)
        # starts from 1
        self.predid = -1
        # starts from 0
        self.tokenid = tokenid
        self.predsense = fields[13]
        self.oracle = ""
        # if experimenting with predicted tags
        if use_predicted:
            self.feat = self.pfeat
            self.pos = self.ppos
            self.lemma = self.plemma
        # proper oracle representation
        infmorph = ""
        if self.feat != "_":
            infmorph = "+"+self.pos+"+"+("+".join(self.feat.split("|")))
        if self.word != "_":
            self.oracle = "word:"+self.word+"+lemma:"+self.lemma+infmorph
        elif self.lemma != "_":
            self.oracle = "+lemma:" + self.lemma+ infmorph+"^DB"
        else:
            self.oracle = infmorph+"^DB"

    def update_token(self, fields):
        # update ids
        self.ids.append(fields[0])
        self.word = fields[1]
        if self.lemma=="_":
            self.lemma = fields[2]
        self.pos = fields[4]
        self.feat = fields[6]
        infmorph = ""
        if self.feat != "_":
            infmorph = "+"+self.pos+"+"+("+".join(self.feat.split("|")))
        elif self.pos != "_":
            infmorph = "+" + self.pos
        if self.word != "_":
            if self.oracle.startswith("+lemma"):
                self.oracle = "word:"+self.word+self.oracle+infmorph
            else:
                self.oracle = "word:"+self.word+"+lemma:"+self.lemma+self.oracle+infmorph
        elif self.lemma != "_":
            self.oracle = "+lemma:"+self.lemma+self.oracle+infmorph+"^DB"
        else:
            self.oracle = infmorph+"^DB"

    def set_predid(self, id):
        self.predid=id
