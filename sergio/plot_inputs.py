import ROOT as r 
r.gROOT.SetBatch(True)
import glob
files=glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/sesanche/SMEFTNet/v6/*.root")
ch=r.TChain("Events")
for k in files:
    ch.Add(k)
c=r.TCanvas()
preselection = "((genJet_pt>500) && (dR_genJet_maxq1q2 < 0.6) && (genJet_SDmass > 70) && (genJet_SDmass < 110))"
target = "TMath::ATan2(parton_hadV_q2_phi-parton_hadV_q1_phi,parton_hadV_q2_eta-parton_hadV_q1_eta)"
ch.Draw("TMath::ATan2(gen_Deta_lab,gen_Dphi_lab)", preselection + '&& (%s < 0.2) && gen_pt_lab > 5'%target, 'norm,same')
for iev,ev in enumerate(ch): 
    if not ((ev.genJet_pt>500) and (ev.dR_genJet_maxq1q2 < 0.6) and (ev.genJet_SDmass > 70) and (ev.genJet_SDmass < 110)): continue
    gr=r.TGraph()
    count=0
    for i in range(ev.ngen):
        if ev.gen_pt_lab[i] < 3: continue
        gr.SetPoint(count, ev.gen_Deta_lab[i],ev.gen_Dphi_lab[i])
        count=count+1

    gr2=r.TGraph()
    gr2.SetPoint(0,
                 ev.parton_hadV_q2_eta-ev.parton_hadV_q1_eta,
                 ev.parton_hadV_q2_phi-ev.parton_hadV_q1_phi)

    print(ev.parton_hadV_q2_eta-ev.parton_hadV_q1_eta,ev.parton_hadV_q2_phi-ev.parton_hadV_q1_phi)

    gr2.SetMarkerStyle(r.kFullCircle)
    gr2.SetMarkerColor(r.kRed)
    gr.SetMarkerStyle(r.kFullCircle)
    frame=r.TH1F("h","",1,-0.8,0.8)
    frame.GetYaxis().SetRangeUser(-0.8,0.8)
    frame.Draw()
    gr.Draw("P,same")
    gr2.Draw("P,same")
    
    c.SaveAs("plot_%d.png"%iev)
    if iev > 100:
        break
#ch.Draw(target, preselection, 'norm,same')

