# -*- coding: utf-8 -*-
"""
Numerical implementation of energy-based homogenization on Miura origami
Written by: Xuwen Li, Amin Jamalimehr

Please run the following codes in Abaqus/CAE using File > Run script and select this python file,
or run in ABAQUS command by typing: abaqus cae noGUI=Energy-basedHomogenization.py
Results are recorded in the text file effectiveConstantsEH.txt
"""

from abaqus import *
import testUtils
testUtils.setBackwardCompatibility()
from abaqusConstants import *
from caeModules import *
import sketch
import part
from regionToolset import*
import load
from shutil import *
from odbAccess import *
import assembly
from itertools import combinations_with_replacement
import itertools
import numpy as np
import math

strain = ['e11','e22','e12','k11','k22','k12']#
Job_Types = []
for (n1,n2) in list(combinations_with_replacement(strain, 2)):
    Job_Types.append(n1+n2)

DOF = [1,2,3]

modelName = 'curvedMiura'
modelid = 'curvedMiura'
## Draw a sketch for each part
theta=np.pi*30/180 #fold angle
gmma = 60*np.pi/180 #sector angle
zeta = np.arctan(np.cos(theta)*np.tan(gmma))
psi = np.arcsin(np.sin(theta)*np.sin(gmma))
panelSideLength = 20.0
a = 20.0
b = 20.0
meshSize = 1.0/16.0*a
H = a*np.sin(theta)*np.sin(gmma)
S = b*np.cos(theta)*np.tan(gmma)/np.sqrt(1+np.cos(theta)**2*np.tan(gmma)**2)
L = a*np.sqrt(1-np.sin(gmma)**2*np.sin(theta)**2)
V = b/np.sqrt(1+np.cos(theta)**2*np.tan(gmma)**2)
x = np.array([0.0, 2.0*S, 0.0, 2.0*S, 0.0, 2.0*S])
y = np.array([0.0, 0.0, L, L, 2.0*L,2.0*L])
z = np.array([0.0, 0.0, H, H, 0.0, 0.0])
curveCoord = np.array([[0.0,0.0,0.0],[S,V,0.0],[2*S,0.0,0.0],
                      [0.0,L,H],[S,L+V,H],[2*S,L,H],
                      [0.0,2*L,0.0],[S,2*L+V,0.0],[2*S,2*L,0.0]])
connect = {0:[0, 1, 3, 2, 0], # Node connectivity
            1:[2, 3, 5, 4, 2]
            }
creases = [[2,3]]
#Find all nodes on the boundaries
FrontN = [1,3,5]
BackN = [0,2,4]
TopN = [4,5]
BottomN = [0,1]

## Assign connectors
vertices = {2:[],
            3:[]}
for vertex in vertices: 
    for panel in connect.keys():
        if vertex in connect[panel]:
            vertices[vertex].append(panel)
    vertices[vertex].sort()# a list of panels connected to the vertex
# Put boundary mesh nodes into groups
Boundaries = {'Back':{'nodes':[],'midpoints':[[0.0,L/2.0,H/2.0,[0,2]],[0.0,1.5*L,H/2.0,[2,4]]],'vertices':BackN,
                      'oppBoundary':'Front','latticeVector':np.array([2*S,0,0])},
              'Bottom':{'nodes':[],'midpoints':[[S,V,0.0,[0,1]]],'vertices':BottomN,
                        'oppBoundary':'Top','latticeVector':np.array([0,2*L,0])}}
groupedBoundaries = Boundaries.keys()

m = mdb.Model(name = modelName)
a = m.rootAssembly
## Material properties
m.Material(name='Mylar')
m.materials['Mylar'].Elastic(table=((4.0e3, 0.38), ))
m.HomogeneousShellSection(name='Section1',
                          material='Mylar',
                          thickness=0.13)

edgeList = [[[0,1],0],
            [[2,3],0],
            [[0,2],0],
            [[1,3],0],
            [[2,3],1],
            [[4,5],1],
            [[2,4],1],
            [[3,5],1]]
for panel in connect.keys():
    p = m.Part(name='Panel'+str(panel), dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    d1 = [] #list of datum points defining vertices
    for n in range(9):
        d1.append(p.DatumPointByCoordinate(coords=(curveCoord[n][0], curveCoord[n][1], curveCoord[n][2])))
    w = [0] * 2 #list of wires defining panel edges
    e=p.edges
    w[0]=p.WireSpline(points=((p.datums[d1[panel*3+0].id], p.datums[d1[panel*3+1].id], p.datums[d1[panel*3+2].id])), mergeType=IMPRINT, 
        meshable=ON, smoothClosedSpline=ON)
    w[1]=p.WireSpline(points=((p.datums[d1[panel*3+3].id], p.datums[d1[panel*3+4].id], p.datums[d1[panel*3+5].id])), mergeType=IMPRINT, 
        meshable=ON, smoothClosedSpline=ON)
    p.ShellLoft(loftsections=((e[0], ), (e[1], )), startCondition=NONE, 
        endCondition=NONE)
    ## Instance
    ins = a.Instance(name='panel_'+str(panel), part=p, dependent=OFF)
ins = a.InstanceFromBooleanMerge(name='UC', instances=(a.instances['panel_0'], 
        a.instances['panel_1'], ), keepIntersections=ON, 
        originalInstances=DELETE, domain=GEOMETRY)
## Section
p = m.parts['UC']
p.SectionAssignment(region=Region(faces=p.faces), sectionName='Section1')
## Mesh
a.makeIndependent(instances=(ins, ))
elem_type = mesh.ElemType(elemCode=S4R)
a.setElementType(regions=(ins.faces,), elemTypes=(elem_type,))
a.seedEdgeBySize(edges=ins.edges, constraint=FIXED, size=meshSize)
a.setMeshControls(regions=ins.faces, elemShape=TRI, allowMapped=ON, technique=FREE)
a.generateMesh(regions=(ins,))
## Step
m.StaticStep(name='Step-1', previous='Initial', maxNumInc=100000, 
    initialInc=0.001, maxInc=0.1)
m.FieldOutputRequest(name='F-Output-3', createStepName='Step-1', variables=('ELSE',))
# Locate midpoints of the creases
midpoints = [[S,V,0.0,[0,1]],[0.0,L/2.0,H/2.0,[0,2]],[2.0*S,L,H/2.0,[1,3]],
              [S,L+V,H,[2,3]],[0.0,1.5*L,H/2.0,[2,4]],[2.0*S,1.5*L,H/2.0,[3,5]],
              [S,2*L+V,0.0,[4,5]]]
## Boundary conditions: unit strain displacement fields
# Create a node list
nodeList = [] # n is the variable for all nodes in the unit cell
nodeLabels = [] # Node labels of all nodes
nseq = []
for ins in a.instances.keys():
    for i in a.instances[ins].nodes:
        nodeList.append(i)
        nodeLabels.append((i.label,ins))
    nseq.append(a.instances[ins].nodes)
N = len(nodeList) # Total number of nodes
# Create a set for all nodes
a.Set(name='nodes',nodes=tuple(nseq))
# Find the outer dimensions of the unit cell
MinX = nodeList[0].coordinates[0]; MaxX = nodeList[0].coordinates[0];
MinY = nodeList[0].coordinates[1]; MaxY = nodeList[0].coordinates[1];
MinZ = nodeList[0].coordinates[2]; MaxZ = nodeList[0].coordinates[2];
for i in range(0,N):

    c = nodeList[i].coordinates

    if c[0] < MinX:
        MinX = c[0]
    if c[0] > MaxX:
        MaxX = c[0]

    if c[1] < MinY:
        MinY = c[1]
    if c[1] > MaxY:
        MaxY = c[1]

    if c[2] < MinZ:
        MinZ = c[2]
    if c[2] > MaxZ:
        MaxZ = c[2]
# Find the back and bottom nodes
for boundary in Boundaries:
    for mp in Boundaries[boundary]['midpoints']:
        for panel in a.instances.keys():
            e = a.instances[panel].edges.findAt((mp[0:3],),)
            if len(e) > 0:
                a.Set(edges=e, name='edge' + str(mp[3][0])+'_'+str(mp[3][1]))
                for n in a.sets['edge' + str(mp[3][0])+'_'+str(mp[3][1])].nodes:
                    # Exclude corner nodes
                    coordn = np.array(n.coordinates)
                    if np.linalg.norm(coordn[0:2]-np.array([x[0],y[0]])) < panelSideLength/200:
                        nodeLoc = 'BottomBack'
                        Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                        a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
                    elif np.linalg.norm(coordn[0:2]-np.array([x[1],y[1]])) < panelSideLength/200:
                        nodeLoc = 'BottomFront'
                        Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                        a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
                    elif np.linalg.norm(coordn[0:2]-np.array([x[4],y[4]])) < panelSideLength/200:
                        nodeLoc = 'TopBack'
                        Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
                        a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)
                    Boundaries[boundary]['nodes'].append(n)
# Find corresponding nodes in the opposite boundaries
for boundary in groupedBoundaries:
    ob = Boundaries[boundary]['oppBoundary']
    Boundaries[ob] = {'nodes':[]}
    for n in Boundaries[boundary]['nodes']:
        oriCoord = np.array(n.coordinates)
        transCoord = oriCoord + Boundaries[boundary]['latticeVector']
        # Find the node at transCoord
        for edgeNode in nodeList:
            c2 = np.array(edgeNode.coordinates)
            if np.linalg.norm(transCoord-c2)<meshSize/4:
                Boundaries[ob]['nodes'].append(edgeNode)
                break
# Find the TopFront corner
for n in a.sets['nodes'].nodes:
    coordn = np.array(n.coordinates)
    if np.linalg.norm(coordn[0:2]-np.array([x[5],y[5]])) < panelSideLength/200:
        nodeLoc = 'TopFront'
        Ri = regionToolset.Region(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)))
        a.Set(nodes=a.instances[n.instanceName].nodes.sequenceFromLabels(labels=(n.label,)), name=nodeLoc)

## Equations (periodic boundary conditions)
for (strain1,strain2) in list(combinations_with_replacement(strain, 2)):
    Type = strain1 + strain2
    for bc in m.boundaryConditions.keys():
        del m.boundaryConditions[bc]
    for pbc in m.constraints.keys():
        del m.constraints[pbc]
    cornerNodes = []
    nodeLocList = ['BottomBack','BottomFront','TopBack','TopFront']
    
    # PBC on edges
    origin = a.sets['BottomBack'].nodes[0].coordinates
    for boundary in ['Back','Bottom']:
        num_nodes = len(Boundaries[boundary]['nodes'])
        for n in range(num_nodes):
            n1 = Boundaries[boundary]['nodes'][n]
            n2 = Boundaries[Boundaries[boundary]['oppBoundary']]['nodes'][n]
            i = 0
            for nodeLoc in ['BottomFront','TopBack','TopFront']:
                if n2 in a.sets[nodeLoc].nodes:
                    n1 = a.sets['BottomBack'].nodes[0]
                    i = 1
            if i == 0: 
                a.Set(nodes=a.instances[n1.instanceName].nodes.sequenceFromLabels(labels=(n1.label,)), name=str(n1.instanceName)+'N'+str(n1.label))
                a.Set(nodes=a.instances[n2.instanceName].nodes.sequenceFromLabels(labels=(n2.label,)), name=str(n2.instanceName)+'N'+str(n2.label))
                x_nb = n1.coordinates[0]-origin[0]
                y_nb = n1.coordinates[1]-origin[1]
                z_nb = n1.coordinates[2]
                x_nf = n2.coordinates[0]-origin[0]
                y_nf = n2.coordinates[1]-origin[1]
                z_nf = n2.coordinates[2]
                strainBC = {'e11':np.array([x_nf-x_nb,0,0]),'e22':np.array([0,y_nf-y_nb,0]),'e12':np.array([(y_nf-y_nb),(x_nf-x_nb),0]),
                            'k11':np.array([(x_nf-x_nb)*((MaxZ+MinZ)/2.0-z_nf),0,(x_nf**2-x_nb**2)/2.0]),
                            'k22':np.array([0,(y_nf-y_nb)*((MaxZ+MinZ)/2.0-z_nf),(y_nf**2-y_nb**2)/2.0]),
                            'k12':np.array([(y_nf-y_nb)*((MaxZ+MinZ)/2.0-z_nf),(x_nf-x_nb)*((MaxZ+MinZ)/2.0-z_nf),(x_nf*y_nf-x_nb*y_nb)])}
                if strain1 == strain2:
                    PBCutheta = strainBC[strain1]
                else:
                    PBCutheta = strainBC[strain1]+strainBC[strain2]
                rp = a.ReferencePoint(point=(3*S+x_nf, y_nf, z_nf))
                rpName = 'RP'+n2.instanceName+str(n2.label)
                rpSet = a.Set(name=rpName, referencePoints=(a.referencePoints[rp.id],))
                tol = panelSideLength/200.0
                
                m.DisplacementBC(name=rpName,createStepName='Step-1',
                                  region=rpSet,u1=PBCutheta[0],u2=PBCutheta[1],u3=PBCutheta[2])
            
                for dof in DOF:
                    m.Equation(name=str(n1.instanceName)+'N'+str(n1.label)+str(n2.instanceName)+'N'+str(n2.label)+'DOF'+str(dof),
                                    terms=((-1.0,str(n1.instanceName)+'N'+str(n1.label),dof),
                                          (1.0,str(n2.instanceName)+'N'+str(n2.label),dof),
                                          (-1.0,rpName,dof)))
    # BC
    m.DisplacementBC(name='BottomBack',createStepName='Step-1',
                      region=a.sets['BottomBack'],u1=0,u2=0,u3=0,ur2=0) 
    ## Job
    job = mdb.Job(name='MiuraPBC-'+Type+modelid,model=modelName,numCpus=4, 
                  numDomains=4, numGPUs=0)
    try:
        job.submit()
        job.waitForCompletion()
    except:
        pass

## Postprocessing
A = np.zeros((6,6))
Auc = 2*S*2*L*H
strainV = np.array([1,1,2,1,1,2])
# Diagonals
for i in range(6):
    Type = strain[i]
    JobName = 'MiuraPBC-'+Type+Type+modelid+'.odb'
    Etemp = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs['ELSE']
    Euc = 0
    for val in Etemp.values:
        Euc = Euc + val.data
    A[i,i] = 2*Euc/Auc/strainV[i]**2
    session.odbs[JobName].close()
A2 = A.copy()
# Off-diagonal terms
for i in list(itertools.combinations(range(6),2)):
    Type = strain[i[0]]+strain[i[1]]
    JobName = 'MiuraPBC-'+Type+modelid+'.odb'
    Etemp = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs['ELSE']
    Euc = 0
    for val in Etemp.values:
        Euc = Euc + val.data
    strainV2 = np.zeros((1,len(strainV)))
    strainV2[0,i[0]] = 1
    strainV2[0,i[1]] = 1
    strainV2 = strainV2*strainV
    A[i[0], i[1]] = 1.0/(2*strainV2[0,i[0]]*strainV2[0,i[1]])*(2.0*Euc/Auc - np.dot(np.dot(strainV2, A2), np.transpose(strainV2)))
    A[i[1],i[0]] = A[i[0],i[1]]
    session.odbs[JobName].close()
# Compute the compliance matrix
SA = np.linalg.inv(A)

outtxt = open('effectiveConstantsEH.txt', 'w+')
outtxt.write('Stiffness matrix ')
outtxt.write(str(A))
outtxt.write('Compliance matrix ')
outtxt.write(str(SA))
outtxt.close()