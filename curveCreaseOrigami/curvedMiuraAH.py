# -*- coding: utf-8 -*-
"""
Numerical implementation of asymptotic homogenization on Miura origami
Written by: Xuwen Li, Amin Jamalimehr

Please run the following codes in Abaqus/CAE using File > Run script and select this python file,
or run in ABAQUS command by typing: abaqus cae noGUI=asymptoticHomogenization.py
Results are recorded in the text file effectiveConstantsAH.txt
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
from itertools import chain
from itertools import combinations_with_replacement
import itertools
import numpy as np
import math

tol = 1e-4
strain = ['e11','e22','e12','k11','k22','k12']#
Job_Types = []
for (n1,n2) in list(combinations_with_replacement(strain, 2)):
    Job_Types.append(n1+n2)
laminaAH = {}
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

def buildUC():        
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
    m.FieldOutputRequest(name='F-Output-3', createStepName='Step-1', variables=('E', 'COORD' ))
    
    # Locate midpoints of the creases
    midpoints = [[S,V,0.0,[0,1]],[0.0,L/2.0,H/2.0,[0,2]],[2.0*S,L,H/2.0,[1,3]],
              [S,L+V,H,[2,3]],[0.0,1.5*L,H/2.0,[2,4]],[2.0*S,1.5*L,H/2.0,[3,5]],
              [S,2*L+V,0.0,[4,5]]]
    crConn = {} #dictionary of creases with their connecting panels
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
    
    return [m, a, nodeList, nodeLabels, N, crConn, edgeList]

##
# Find nodal reactions of unit strain displacement fields
##
def disp_assigner(label,JobName,BCname):
    Translational_DOF = [1,2,3]
    Rotational_DOF    = [4,5,6]
    setName =  'C'+ Type + str(label[1]) + 'N' + str(label[0])
    BCRegion = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
    Value = np.zeros(6)
    if BCname=='CD':
        ufield = session.openOdb(name='Uniform-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
        usubset = session.odbs['Uniform-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
        cfield = session.openOdb(name='Characteristic-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
        csubset = session.odbs['Characteristic-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
        for dof in DOF:
            if dof in Translational_DOF:
                Value[dof-1] = ufield['U'].getSubset(region=usubset).values[label[0]-1].data[dof-1] + cfield['U'].getSubset(region=csubset).values[label[0]-1].data[dof-1]
            if dof in Rotational_DOF:
                Value[dof-1] = ufield['UR'].getSubset(region=usubset).values[label[0]-1].data[dof-4] + cfield['UR'].getSubset(region=csubset).values[label[0]-1].data[dof-4]
    else:
        field = session.openOdb(name='UnitStrain-'+JobName+'.odb').steps['Step-1'].frames[-1].fieldOutputs
        subset = session.odbs['UnitStrain-'+JobName+'.odb'].rootAssembly.instances[str(label[1]).upper()]
        for dof in DOF:
            if dof in Translational_DOF:
                Value[dof-1] = field['U'].getSubset(region=subset).values[label[0]-1].data[dof-1]
            if dof in Rotational_DOF:
                Value[dof-1] = field['UR'].getSubset(region=subset).values[label[0]-1].data[dof-4]
    m.DisplacementBC(name=BCname+Type+setName,
                  createStepName='Step-1',region=BCRegion,
                  u1=Value[0],u2=Value[1],u3=Value[2])#,ur1=Value[3],ur2=Value[4],ur3=Value[5]

[m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
## Boundary conditions: unit strain displacement fields
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
## Compute the unit strain displacement fields
for (strain1,strain2) in list(combinations_with_replacement(strain, 2)):
    Type = strain1 + strain2
    JobName = Type+modelid
    # supress all connectors and PBCs
    for bc in m.boundaryConditions.keys():
        del m.boundaryConditions[bc]
    # # assign unit strain displacement fields
    for i in nodeList:
        X = i.coordinates[0] - MinX
        Y = i.coordinates[1] - MinY
        Z = i.coordinates[2] - (MinZ+MaxZ)/2.0
        Ri = regionToolset.Region(nodes=a.instances[i.instanceName].nodes.sequenceFromLabels(labels=(i.label,)))
        strainBC = {'e11':np.array([float(X),0.0,0.0]),
                    'e22':np.array([0.0, float(Y),0.0]),
                    'e12':np.array([float(0.5*(Y)), float(0.5*(X)), 0.0]),
                    'k11':np.array([-float(Z*X),0.0,float(X**2/2.0)]),
                    'k22':np.array([0.0,-float(Z*Y),float(Y**2/2.0)]),
                    'k12':np.array([-float(Z*Y/2.0),-float(Z*X/2.0),float(X*Y/2.0)])}
        if strain1 == strain2:
            UnitStrainU = strainBC[strain1]
        else:
            UnitStrainU = strainBC[strain1] + strainBC[strain2]
        m.DisplacementBC(name='US'+Type+str(i.label)+i.instanceName,
                          createStepName='Step-1',region=Ri,
                          u1=UnitStrainU[0],u2=UnitStrainU[1],u3=UnitStrainU[2])
    ## Job
    job = mdb.Job(name='Uniform-'+Type+modelid,model=modelName,numCpus=4, 
                  numDomains=4, numGPUs=0)
    try:
        job.submit()
        job.waitForCompletion()
    except:
        pass

# # Write the uniform X results into a txt file using the function
fields = {}
for Type in Job_Types:
    JobName = 'Uniform-'+Type+modelid+'.odb'
    fields[Type] = session.openOdb(name=JobName).steps['Step-1'].frames[-1].fieldOutputs # Results of the last frame in the first step
    
##
# Find characteristic displacement fields
##
[m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
#Find all edges on the boundaries
boundaries = []
for panel in connect:
    for n in range(4):
        edge = [connect[panel][n],connect[panel][n+1]]
        for c in creases:
            if set(edge)!=set(c):
                boundaries.append(edge)
connSet = []
connList = []
# Find the back and bottom nodes
for boundary in Boundaries:
    for mp in Boundaries[boundary]['midpoints']:
        for panel in a.instances.keys():
            e = a.instances[panel].edges.findAt((mp[0:3],),)
            if len(e) > 0:
                a.Set(edges=e, name='edge' + str(mp[3][0])+'_'+str(mp[3][1]))
                for n in a.sets['edge' + str(mp[3][0])+'_'+str(mp[3][1])].nodes:
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
            
PBCedges = {}
for boundaryCombinations in list(itertools.combinations(groupedBoundaries,2)):
    b1nodes = Boundaries[boundaryCombinations[0]]['nodes']
    b2nodes = Boundaries[boundaryCombinations[1]]['nodes']
    edgeName = (boundaryCombinations[0],boundaryCombinations[1])
    edgelist = [ [] for i in range(4)]
    for n1 in b1nodes:
        if n1 in b2nodes:
            edgelist[0].append(n1)
            c1 = []
            c1.append(np.array(n1.coordinates))
            lv1 = Boundaries[boundaryCombinations[0]]['latticeVector']
            lv2 = Boundaries[boundaryCombinations[1]]['latticeVector']
            c1.append(c1[0] + lv1)
            c1.append(c1[0] + lv2)
            c1.append(c1[0] + lv1 + lv2)
            for i in range(1,len(c1)):
                for n2 in nodeList:
                    c2 = np.array(n2.coordinates)
                    if np.linalg.norm(c1[i]-c2)<0.01:
                        edgelist[i].append(n2)
                        break
    PBCedges[edgeName]=edgelist           
flat_PBCedges = [item2 for sublist in PBCedges.values() for item1 in sublist for item2 in item1]
## Equations (periodic boundary conditions)
def write_equations(n1,n2):
    a.Set(nodes=a.instances[n1.instanceName].nodes.sequenceFromLabels(labels=(n1.label,)), name=str(n1.instanceName)+'N'+str(n1.label))
    a.Set(nodes=a.instances[n2.instanceName].nodes.sequenceFromLabels(labels=(n2.label,)), name=str(n2.instanceName)+'N'+str(n2.label))
    for dof in DOF:
        m.Equation(name=str(n1.instanceName)+'N'+str(n1.label)+str(n2.instanceName)+'N'+str(n2.label)+'DOF'+str(dof),
                    terms=((1.0,str(n1.instanceName)+'N'+str(n1.label),dof),
                          (-1.0,str(n2.instanceName)+'N'+str(n2.label),dof)))
        
# Interior nodes of boundary faces
for boundary in groupedBoundaries:
    num_nodes = len(Boundaries[boundary]['nodes'])
    for n in range(num_nodes):
        n1 = Boundaries[boundary]['nodes'][n]
        n2 = Boundaries[Boundaries[boundary]['oppBoundary']]['nodes'][n]
        c1 = np.array(n1.coordinates)
        PBCedgeNode = 0
        for pbcedgenode in flat_PBCedges:
            if n1 == pbcedgenode:
                PBCedgeNode = 1
        if PBCedgeNode: # skip all boundary edges
            continue
        write_equations(n1,n2)
# Boundary edges
for edgeName in PBCedges:
    if PBCedges[edgeName][0]:
        for n in range(len(PBCedges[edgeName][0])):
            for edge in range(len(PBCedges[edgeName])-1):
                n1 = PBCedges[edgeName][edge][n]
                n2 = PBCedges[edgeName][edge+1][n]
                write_equations(n1,n2)

## Boundary condition preventing rigid body motions
for bc in m.boundaryConditions.keys():
    del m.boundaryConditions[bc]
for label in nodeLabels:
    Set = str(label[1])+'N'+str(label[0])
    if Set in list(chain.from_iterable(connList)): # Skip assigning BC if a node is at a connector
        continue
    elif Set in a.sets.keys(): # Skip assigning BC if a node is at the unit cell boundary
        continue
    else:
        break
fixedBC = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
m.EncastreBC(name='fixedBC', createStepName='Initial', 
    region=fixedBC, localCsys=None)

## Loads: apply reaction forces from the unit strain displacement field
# A function to assign loads
def load_assigner(label,field):
    Translational_DOF = [1,2,3]
    Rotational_DOF    = [4,5,6]
    setName =  'C'+ Type + str(label[1]) + 'N' + str(label[0])
    loadRegion = regionToolset.Region(nodes=a.instances[label[1]].nodes.sequenceFromLabels(labels=(label[0],)))
    Value = np.zeros(6)
    for dof in DOF:
        
        if dof in Translational_DOF:
            subset = session.odbs['Uniform-'+Type+modelid+'.odb'].rootAssembly.instances[str(label[1]).upper()]
            Value[dof-1] = - field['RF'].getSubset(region=subset).values[label[0]-1].data[dof-1]
            
        if dof in Rotational_DOF:
            subset = session.odbs['Uniform-'+Type+modelid+'.odb'].rootAssembly.instances[str(label[1]).upper()]
            Value[dof-1] = - field['RM'].getSubset(region=subset).values[label[0]-1].data[dof-4]
        
    if np.linalg.norm(Value[0:3]) > tol:
        m.ConcentratedForce(name=setName+'cf', createStepName='Step-1', 
                                region=loadRegion, cf1=Value[0], cf2=Value[1], cf3=Value[2], distributionType=UNIFORM, field='', 
                                localCsys=None)
    
for Type in Job_Types:
    for cf in m.loads.keys():
        m.loads[cf].suppress() 
    for label in nodeLabels:
        load_assigner(label,fields[Type])
    ## Job
    job = mdb.Job(name='Characteristic-'+Type+modelid,model=modelName,numCpus=4, 
                  numDomains=4, numGPUs=0)
    try:
        job.submit()
        job.waitForCompletion()
    except:
        pass  

##
# Find difference between uniform and characteristic displacement fields
##
[m, a, nodeList, nodeLabels, N, crConn, edgeList] = buildUC()
m.FieldOutputRequest(name='F-Output-else', createStepName='Step-1', variables=('ELSE',))
for Type in Job_Types:
    JobName = Type+modelid
    # assign displacement field
    for bc in m.boundaryConditions.keys():
        m.boundaryConditions[bc].suppress()
    for label in nodeLabels:
        disp_assigner(label,JobName,'CD')
    ## Job
    job = mdb.Job(name='Diff-'+Type+modelid,model=modelName,numCpus=4, 
                  numDomains=4, numGPUs=0)
    try:
        job.submit()
        job.waitForCompletion()
    except:
        pass

## Postprocessing
A = np.zeros((6,6))
Auc = 2*S*2*L*H
strainV = np.array([1,1,1,1,1,1])
# Diagonals
for i in range(6):
    Type = strain[i]
    JobName = 'Diff-'+Type+Type+modelid+'.odb'
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
    JobName = 'Diff-'+Type+modelid+'.odb'
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
    for ok in session.odbs.keys():
        session.odbs[ok].close()
# Compute the compliance matrix
SA = np.linalg.inv(A)

outtxt = open('effectiveConstantsAH.txt', 'w+')
outtxt.write('Stiffness matrix ')
outtxt.write(str(A))
outtxt.write('Compliance matrix ')
outtxt.write(str(SA))
outtxt.close()