# -*- coding: utf-8 -*-
"""
2-D cellular automata to mimic camouflage using an 
Activator and Inhibitor model.  The boundaries are wrapped.


@author: Peter Schwartz
"""
import math
#import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors

def setupCA():
    ## setup random initial configuration of CA
    li = [-1,1]
    dim = 30
    
    CA = np.array([[random.choice(li) for j in range(0,dim)] for i in range(0,dim)])
    
    return CA

def AIsums(ca,i1,j1,R1,R2,J1,J2):
    ca_t = np.copy(ca) #ensure ca isn't changed here
    dim = ca_t.shape[0]
    #indices = [ [dist(i1,j1,i,j,dim) >= R1 and dist(i1,j1,i,j,dim) < R2  for j in range(0,dim)] for i in range(0,dim)]
    
    #find all cells within < R1 and sum them
    inner_nbd = [ca_t[i,j] for j in range(0,dim) 
                 for i in range(0,dim)  if dist(i1,j1,i,j,dim) < R1 ]
    

    inner_sum = float(J1*sum(inner_nbd))
    
    # find all cells within R1<= d < R2 and sum them

    outer_nbd = [ca_t[i,j] for j in range(0,dim) for i in range(0,dim) 
                 if dist(i1,j1,i,j,dim) >= R1 and dist(i1,j1,i,j,dim) < R2]
    
    outer_sum = float(J2*sum(outer_nbd));
    
    return outer_sum + inner_sum;


        
def dist(i1,j1,i,j, dim):
    xdiff = min((i1-i)%dim,(i-i1)%dim)
    ydiff = min((j1-j)%dim,(j-j1)%dim)
    
    return xdiff + ydiff;


def evolveAICA(ca,uporder,bias,R1,R2,J1,J2):
    
    
    stable = False  #flag to determine when CA is stationary
    while(not stable):
        old = np.copy(ca)
        #shuffle update order
        random.shuffle(update_order)
        
        #update according to new random order!
        for el in uporder:
            i1 = el[0] #row of element to be updated
            j1 = el[1] #col of element to be updated
            
            ca[i1,j1] = np.sign(bias + AIsums(ca,i1,j1,R1,R2,J1,J2) )
        
        
        #check if all cells have converged:
        test = ca - old
        numzeros = np.count_nonzero(test) 
        if numzeros == 0:
            stable = True #triggered if any_zeros is empty
        print("plotting update","num_nonzeros:",numzeros)
        #plotCA(ca)

def plotCA(ca):
    data = np.copy(ca)
    states = ['black','white']
    cmap = colors.ListedColormap(states)
    plt.figure()
    ax = plt.gca()
    ax.imshow(data,cmap=cmap)
    plt.title("J1 = %.2f J2 = %.2f; R1= %i, R2= %i, bias= %.f" %(J1,J2,R1,R2,bias))
    
    if step == 0: 
        plt.savefig("J1="+str(J1)+"J2="+str(J2)
                    +"R1=" + str(R1) + "R2="+str(R2)+"h="+str(bias)+".pdf",
                              format='pdf',bbox_inches='tight')
    
    
    plt.show()
    
    
def spatial_correlation_and_entropy(ca):
    cat = np.copy(ca)
    dim = cat.shape[0]
    rho = []; #holds the spatial correlation for each distance, l
    H_l = []; #holds entropy for each distance, l
    I_l = []; #holds avg mutual information, l
    H = 0.0; #overall entropy
    
    #calculate self_correltation:
    self_correlation = float(np.sum(cat))/float(dim**2);
    self_correlation = self_correlation**2
    
    #calculate for l = 0;
    rho.append(abs(1.0-self_correlation))
    indx_arr = range(0,dim);
    
    #get number of occurences of state = +1
    numofones = (cat == 1).sum() #sum(x.count(1) for x in cat)
    prPlus = float(numofones)/float(dim**2)
    
    if prPlus != 0.0 and prPlus != 1.0: 
        print(prPlus)
        H = -1.0*(prPlus*math.log2(prPlus)+(1.0-prPlus)*math.log2(1.0-prPlus))
    
    elif prPlus == 1.0:
        H = -1.0*(prPlus*math.log2(prPlus))
    
    elif prPlus == 0.0:
        H = -1.0*((1.0-prPlus)*math.log2(1.0-prPlus))
    
    
    
    for l in range(1,int(dim/2-1)+1):
        #hold the running total for length l
        
        rho_sum = 0.0
        prob_pp = 0.0 #will hold probability of +1+1 pair
        prob_mm = 0.0 # will hold probability of -1-1 pair
        
        for i1 in indx_arr:  #row index of cell i (i1,j1)
            for i2 in indx_arr: #row index of cell j (i2,j2)
                
                
                #if i1 != i2:  #if the cells are not in the same row no restriction
                   
                #calculate neighbor for correlation length and store cell states for 
                #calculating Pr(+1,+1) and Pr(-1,-1)
                nbd = [[cat[i1,j1]*cat[i2,j2],cat[i1,j1],cat[i2,j2]] for j1 in indx_arr 
                       for j2 in indx_arr if dist(i1,j1,i2,j2,dim) == l]
                
                
                column=0
                rho_sum += sum(row[column] for row in nbd)
                
                #list that holds all the pairs that are +1,+1
                nbdplus = [s[0] for s in nbd if s[1] == 1 and s[2] == 1]
                #list that holds all the pairs that ar -1,-1
                nbdminus = [s[0] for s in nbd if s[1] == -1 and s[2] == -1]
                
                prob_pp += len(nbdplus)
                prob_mm += len(nbdminus)
            
        
        
        #calculate total correlation for given l
        corr = 1.0/float(dim**2*4.0*l)*rho_sum - self_correlation 
        #append rho_l
        rho.append(abs(corr))
        
        #adjust the probabilities for double counting:
        if prob_pp%2 != 0: print("error: not even")
        if prob_mm%2 != 0: print("error:not even")
        
        prob_pp = 1.0*prob_pp/float(dim**2*4.0*l)
        prob_mm = 1.0*prob_mm/float(dim**2*4.0*l)
        
        prob_mp = 1.0 - prob_pp - prob_mm
        
        tempH = 0.0
        if prob_pp != 0.0:
            tempH+= prob_pp*math.log2(prob_pp)
        if prob_mm != 0.0:
            tempH+= prob_mm*math.log2(prob_mm)
        
        if prob_mp != 0.0:
            tempH+= prob_mp*math.log2(prob_mp)
        
        tempH = -1.0*tempH
        
        H_l.append(tempH)
        
        I_l.append(2.0*H-tempH)
            
    return rho, H_l, I_l;

#%%  Running Experiments
    
## set interaction coeffificents must change this for different experiments
J1 = 1.0;
J2 = -0.1;
#experiment1Data = []
#arrays for experiment 3
R1arr = [1,1,3,3]
R2arr = [9,9,9,9]
#harr =  [0,-4,-2,0,-6,-3,0,0,-1,0,-6,-3,0,0,0,0,0]
harr = [3,6,3,6]
# #arrays for experiment 2
# R1arr = [1,1,1,1,1,1,1,1,1,4,4,4,4,4,9,9,9]
# R2arr = [2,4,4,4,6,6,6,9,13,5,7,7,7,12,12,12,12]
# harr = [0,-2,-1,0,-5,-3,0,0,0,0,-5,-3,0,0,-6,-3,0]

#arrays for experiment 1
# R1arr =[6,6,6]
# R2arr = [15, 15, 15]
# harr = [-1.0,-2.0,-3.0]

for run in range(0,len(R1arr)):
    
    spat_corr = [];
    mutinfo = [];
    corr_length =[];

    for step in range(0,3):
        ## setup random initial configuration of CA
        CA = setupCA()
        dim = CA.shape[0]
        update_order = [];
        update_order = [ [i,j] for i in range(0,dim) for j in range(0,dim)]

        random.shuffle(update_order)



        R1 = R1arr[run]
        R2 = R2arr[run]

        bias = harr[run]

    #     print("plotting initial state")
    #     plotCA(CA)

        evolveAICA(ca=CA,uporder=update_order,bias=bias,R1=R1,R2=R2,J1=J1,J2=J2)

    #     print("plotting finial state")
        plotCA(CA)


        print("calculating spatial correlation and entropy")
        temp_spat_corr, tempHl, temp_mutinfo = spatial_correlation_and_entropy(ca=CA)

        rho0over_e = float(temp_spat_corr[0]/math.exp(1))
        temp_corr_length = min(enumerate(temp_spat_corr), key=lambda x: abs(x[1]-rho0over_e))

        spat_corr.append(temp_spat_corr)
        mutinfo.append(temp_mutinfo)
        corr_length.append(temp_corr_length[0])


    spat_corr_avg = [];

    for col in range(0,15):
        avg = round(sum(row[col] for row in spat_corr)/3.0,3)
        spat_corr_avg.append(avg)
    print("")
    print(spat_corr_avg)

    mutinfo_avg =[];
    for col in range(1,15):
        avg = round(sum(row[col-1] for row in mutinfo)/3.0,3)
        mutinfo_avg.append(avg)

    print("")
    print(mutinfo_avg)

    corr_len_avg = [];
    avg = sum(corr_length)/3.0
    corr_len_avg.append(avg)
    print("")
    print(corr_len_avg)

  #  experiment3Data.append([J1,J2,R1,R2,bias,spat_corr_avg,mutinfo_avg,corr_len_avg])


#csvfile = "project2.csv"
#
#with open(csvfile, "a") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(experiment3data[18:])