from app.utils.nlp.spacy_init import process_text
from app.utils.nlp.keyword_extractor import extract_keywords_descriptors, merge_keyword_dicts
from fastapi import APIRouter

sample_transcript = """
Doctor: Hello? Hi. Um, should we start? Yeah, okay. Hello how um. Good morning sir, how can I help you this morning?
Patient: Hello, how are you?
Patient: Oh hey, um, I've just had some diarrhea for the last three days, um, and it's been affecting me I need to stay close to the toilet. And, um, yeah, it's been affecting my day-to-day activities.
Doctor: Sorry to hear that. Um, and and when you say diarrhea, what'd you mean by diarrhea? Do you mean you're going to the toilet more often? Or are your stools more loose?
Patient: Yeah, so it's like loose and watery stool, going to the toilet quite often, uh and like some pain in my, like, lower stomach?
Doctor: 
Doctor: Okay. And how many times a day are you going, let's say, in the last couple of days?
Patient: Um, probably like six or seven times a day? Yeah.
Doctor: Six, seven times a day. And you mention it's mainly watery. Have you noticed any other things, like blood in your stools?
Patient: No, no blood, yeah, just watery and loose stool.
Doctor: Okay. And you mentioned you've had some pain in your tummy as well. Whereabouts is the pain, exactly?
Patient: Yep.
Patient: So in my lower abdomen, so, uh, like, um...yeah, just to one side.
Doctor: One side. And what side is that?
Patient: Uh, on the left side.
Doctor: Left side. Okay, and can you describe the pain to me?
Patient: Yeah, it feels, um, like a cramp, like a muscular cramp, and, um, yeah i feel a bit uh weak and shaky.
Doctor: Okay. And is the pain, is that, is it there all the time, or does it come and go?
Patient: Uh, it comes and goes.
Doctor: Come and go. Does the pain move anywhere else, for example towards your back?
Patient: Uh...no, just maybe my stomach.
Doctor: Okay, fine. And you mentioned you've been feeling quite weak and shaky as well. What do you mean by shaky? Do you mean you've been having, uh have you been feeling feverish, for example?
Patient: Yeah.
Patient: Um, yeah, it doesn't feel like -- yeah, it just makes me feel weak. I haven't had a fever, um, at the moment, but I did notice um a temperature when the symptoms started, so, um, yeah around about three or four days ago.
Doctor: 
Doctor: You measure your temperature then?
Patient: Yeah, I uh I didn't mention my temperature, no, but I felt, um, just a bit hot. And, y'know.
Doctor: Okay. Okay. Any other symptoms like sweating, or um, night sweats? No? And, uh, any vomiting at all?
Patient: Uh, no.
Patient: Yeah, so um, I vomited at the start of the symptoms but now um I've stopped vomiting.
Doctor: You stopped vomiting, okay. And was your vomit, I know it's not a nice thing to talk about, but was it just normal food colour Yeah. And there was no blood in your vomit, is that right?
Patient: Yeah, yeah, just normal vomit, yeah. No no blood, no. Yeah.
Doctor: No, okay. Um, and um, any any other symptoms at all? So you mentioned tummy pain, you mentioned diarrhea, you mentioned your vomiting, uh, anything else that comes to mind?
Patient: Yep. Um, I had a loss of appetite, um, so I haven't been eating as much, but I've been able to hold down fluids.
Doctor: Okay.
Doctor: Okay, so you're drinking fluids. Um, what kind of foods have you managed to eat, if anything?
Patient: Yep.
Patient: Um, just soups, and, uh, yeah, light foods. Like smoothies and, yeah, liquid foods mainly.
Doctor: Okay. Fine. Um, and sir these started three days ago the symptoms. Are you aware of any triggers which may have caused the symptoms uh to kick on. So for example, think like takeaway foods or eating out or being around other people with similar symptoms.
Patient: 
Patient: Yeah, so I had takeaway about four days ago, um, uh, but other than that I've, yeah, been, uh, eating normally. Nothing unusual here.
Doctor: Okay.
Doctor: Do you remember where you ate?
Patient: Um, yeah, I ate at a Chinese restaurant with friends. Yeah.
Doctor: Okay. Anyone else unwell with similar symptoms?
Patient: Um, so no one else in the family, so a wife and two kids and one, um, child was vomiting, but they haven't got diarrhea. There's no one with the same symptoms.
Doctor: Okay, okay. Fine. Um, alright. And uh, in terms of your , your overall health, are you normally fit and well? Or, uh
Patient: Um, yeah, I mean, other than um athsma, um I use an inhaler, everything uh else is fine.
Doctor: Okay. And, is your asthma well-controlled?
Patient: Uh, yeah, that's fine. I just, yeah, use an inhaler, and uh that's under control.
Doctor: Fine. And you don't have any other tummy problem, bowel problems I should be aware of?
Patient: No.
Doctor: No, okay. Um, and apart from the inhalers, do you take any other medications?
Patient: Uh, no, no other medications.
Doctor: Okay, fine. And in terms of just your day to day life, you said it's been affecting your life, um, in what way has it been affecting your life?
Patient: Yeah.
Patient: Uh, so, I need to stay close to the toilet 'cause I go quite frequently during the these past three days. Um, yeah, other than that, it's uh, yeah, the main concern.
Doctor: Okay.
Doctor: Yeah.
Doctor: And have you, are you currently working at the moment?
Patient: Uh, yes, yeah. I I work, er. Um, I'm an accountant.
Doctor: Would, would work.
Doctor: Okay. Have you been going into work the last three days, or have you been at home?
Patient: Uh, yeah, I've been going to work. Yeah. Yeah, it's been quite difficult.
Doctor: okay. That must be difficult for you then.
Doctor: fine. And you said, you mentioned you live with your wife and two children, is that right?
Patient: Yes, yeah.
Doctor: Right, alright. Um, just a couple of other question we need to ask, sir. Um, do you smoke at all?
Patient: Uh, no, I don't smoke.
Doctor: And do you drink much in the way of alcohol?
Patient: Uh, no, I I don't drink alcohol, no.
Doctor: Okay. so um, er normally at this stage I like to um, examine you if that's okay, but um, um, but but having listened to your story, sir, I think uh, um, just to recap for the last three days you've been having loose stool, diarrhea, a bit of tummy pain uh mainly on the left-hand side, um and vomiting and fever and you're quite weak and lethargic um, you mentioned you had this Chinese takeaway as little as three days ago and I wondered whether that might be the cause of your problems.
Patient: Yeah.
Patient: 
Patient: Okay.
Doctor: Um, it seems like you may have something, uh, called gastroenteritis, which essentially just a tummy bug or infection of your uh of your tummy.
Patient: 
Doctor: Uh, mainly caused by viruses but there can be a possibility of bacteria uh causing its symptoms. Um.
Patient: Yeah.
Patient: Yeah.
Doctor: At this stage, uh, what, what we'd recommend is just what we say conservative management. So, um, I don't think you need anything like antibiotics. It's really just, um, making sure you're well hydrated, so drinking fluids.
Patient: 
Patient: Mm-hmm.
Doctor: Um, there are things like Dioralyte you can get from the pharmacy, which uh it's um it helps helps replenish your minerals and vitamins.
Patient: Okay.
Doctor: Um, and if you are having vomiting diarrhea I would say recommend that in the first, you know, first couple of days.
Patient: Yep.
Doctor: If you are feeling feverish and weak, eh taking some paracetamol, uh, two tablets up to four times a day for the first few days can also help.
Patient: Yep.
Doctor: I will certainly advise you to take some time off work, actually I know you're quite keen to work but I would say the next two, two to three days as the infection clears from your system to take some time off and rest.
Patient: Okay.
Patient: Yeah.
Doctor: Um, I'll admit if your symptoms haven't got better, you know, in in three to four days, I'd like to come and see you again.
Patient: Okay, sure.
Doctor: Because if it is ongoing then we have to wonder whether something else caused your symptoms.
Patient: Yep.
Doctor: Uh, and we may need to do further tests like um taking a sample of your stool so we can test that.
Patient: 
Doctor: Um, etcetera etcetera.
Patient: Yep, sure, yep.
Doctor: How's that sound?
Patient: That sounds great, yeah. Yeah.
Doctor: Do you have any questions for me?
Patient: Um, no, no further questions, no.
Doctor: Okay, and is uh is the treatment plan clear?
Patient: Uh, yes, yeah, that's that's very clear. Thank you.
Doctor: Great. Well, I wish you all the best.
Patient: Okay, thank you. Bye.
Doctor: Thank you. Bye bye.
"""

router = APIRouter()

@router.get("/")
def test_with_sample_transcript():
    transcription_doc = process_text(sample_transcript)
    print(transcription_doc.ents)
    keyword_dicts = extract_keywords_descriptors(doc=transcription_doc)
    print(keyword_dicts)
    result_dicts = []
    for keyword_dict in keyword_dicts:
        if len(result_dicts) == 0:
            result_dicts.append(keyword_dict)
        for final_dict in result_dicts:
            if keyword_dict["term"] == final_dict["term"]:
                result_dicts.append(merge_keyword_dicts(keyword_dict, final_dict))
    print(result_dicts)
    return result_dicts