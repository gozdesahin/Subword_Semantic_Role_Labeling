#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Gözde Gül Şahin

Write predicted semantic roles into a CoNLL-09 SRL file
writeCoNLLUD for Finnish, writeCoNLL for other languages
Writes predicted predicate senses if given

Used by test and ensemble files - only during testing
"""

def writeCoNLL(outFile, ldr, lst, psenseSents=None):
    """
    Write predictions to CONLL file
    :param outFile: file to write
    :param ldr: loader object
    :param lst: predicted semantic roles
    :param psenseSents: predicted predicate senses
    :return:
    """
    labix = 0
    FIXED_COL_CNT = 14
    if psenseSents==None:
        psenseSents = [None]*len(ldr.c9sents)
    for c9sent, psenseSent in zip(ldr.c9sents, psenseSents):
        tokens = c9sent.tokens
        finSentLabs = []
        for i in range(c9sent.predcnt):
            finPredLabs = []
            labsForPred = lst[labix].split(" ")
            for srole, tok in zip(labsForPred, tokens):
                if len(tok.ids)>1:
                    for x in range(len(tok.ids)-1):
                        finPredLabs.append(u"_")
                finPredLabs.append(srole)
            finSentLabs.append(finPredLabs)
            labix+=1
        # sentence rows
        sentRows = ['']*len(c9sent.orgRows)
        for a in range(len(c9sent.orgRows)):
            sentRows[a] = []
            for b in range(FIXED_COL_CNT):
                sentRows[a].append(c9sent.orgRows[a][b])
            if psenseSent!=None:
                sentRows[a][FIXED_COL_CNT-1]=psenseSent.orgRows[a][FIXED_COL_CNT-1]
        for predLabs in finSentLabs:
            for a,lab in zip(range(len(c9sent.orgRows)),predLabs):
                sentRows[a].append(lab)
        # write to file
        for wrrows in sentRows:
            wr = u"\t".join(wrrows)
            outFile.write(wr)
            outFile.write(u"\n")
        outFile.write(u"\n")
    outFile.close()

def writeCoNLLUD(outFile, ldr, lst, psenseSents=None):
    """
    Write predictions to CONLL-UD file - only for Finnish
    :param outFile: file to write
    :param ldr: loader object
    :param lst: predicted semantic roles
    :param psenseSents: predicted predicate senses
    :return:
    """
    labix = 0
    if psenseSents==None:
        psenseSents = [None]*len(ldr.c9sents)
    for c9sent, psenseSent in zip(ldr.c9sents, psenseSents):
        tokens = c9sent.tokens
        finSentLabs = []
        for i in range(c9sent.predcnt):
            finPredLabs = []
            labsForPred = lst[labix].split(" ")
            for srole, tok in zip(labsForPred, tokens):
                finPredLabs.append(srole)
            finSentLabs.append(finPredLabs)
            labix+=1
        # sentence rows
        sentRows = []
        for tok in c9sent.tokens:
            row = []
            row.append(str(tok.id))
            row.append(tok.word)
            row.append(tok.lemma)
            row.append(tok.lemma)
            row.append(tok.pos)
            row.append(tok.ppos)
            row.append(tok.feat)
            row.append(tok.feat)
            row.append(tok.head)
            row.append(tok.head)
            row.append(tok.deplab)
            row.append(tok.deplab)
            if tok.ispred:
                row.append(u"Y")
            else:
                row.append(u"_")
            row.append(tok.predsense)
            sentRows.append(row)

        for predLabs in finSentLabs:
            for a,lab in zip(range(len(c9sent.tokens)),predLabs):
                sentRows[a].append(lab)
        # write to file
        for wrrows in sentRows:
            wr = u"\t".join(wrrows)
            outFile.write(wr)
            outFile.write(u"\n")
        outFile.write(u"\n")
    outFile.close()
