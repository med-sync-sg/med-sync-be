from app.utils.nlp.spacy_utils import process_text
from app.utils.nlp.keyword_extractor import extract_keywords_descriptors, merge_keyword_dicts
from fastapi import APIRouter
import logging

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
sample_transcript_2 = '''
    Doctor: Alex. Ohh. Hello? Hi, can you hear me?
Patient: 
Patient: Yeah.
Doctor: OK great. Um how can I help you this morning, sir?
Patient: Alright, so I've been feeling, I've been feeling kind of uh under the weather for the past four days.
Doctor: 
Patient: Um it started with the um, uh sore throat and runny nose, and
Patient: It's sort of, um, it's sort of going into a cough now. Um, the sort throat is kind of going away, but, um, I'm starting to cough.
Doctor: Mm-hmm. OK.
Patient: And uh kind of uh bad and tired.
Doctor: Sorry to hear that .
Doctor: Mm. OK. I'm sorry to hear that. Seems like there's a lot going on there. Um so let's start with your, your sore throat first. Um tell me a bit more about that, if you can.
Patient: Um ,so I, I , you know I think I, it all started with, uh, this colleague of mine, she just kept sneezing next to me, all the time. I think her kids are catching something from the
Patient: uh from their kindergarten or something. And uh, you know the it it started as normal sore throat, and uh was quite painful for a couple days, but then it was past.
Doctor: OK.
Doctor: OK. OK. And did you have any uh difficulty or pain on swallowing?
Patient: Um yeah. Yeah it was uh, it was quite painful to swallow, for one or two days.
Doctor: OK.
Doctor: You , did you manage to have a look at the back of your throat in the mirror?
Patient: Um no, not really.
Doctor: No. OK you didn't, you didn't notice any abnormal white spots, redness at the back of your throat?
Patient: Um
Patient: No. Um I I didn't think about uh .
Doctor: 
Doctor: that's OK. That's OK. And you mentioned a runny nose. Um again what kind of discharge is coming out? Is it clear, or is it a bit more coloured?
Patient: Yeah yeah, it's uh it's clearing up .
Patient: Sort of um, sort of getting my nose stuffed all the time. It's very had to, uh to get it free.
Doctor: OK. You feel quite congested, do you?
Patient: Yeah, quite congested.
Doctor: OK, fine. Um you also mentioned a cough as well. Can you tell me a bit more about that?
Patient: Um yeah that started recently, like just uh um maybe yesterday. And um it's uh, it's dry for now, it's it's not very painful but it's sort of there, all the time.
Doctor: OK.
Doctor: OK. And is it worse any particular time of the day?
Patient: Sorry?
Doctor: Is it, is it worse any particular time of the day? For example in the evenings, night time, mornings?
Patient: Um
Doctor: throughout the day.
Patient: No it's it's um, it's sort of, it's sort of constant. Um I forgot to mention, that um I'm also having uh night sweats.
Doctor: OK.
Doctor: Night first, OK.
Patient: That's, that's very odd actually. makes me a bit tired, um makes my uneasy.
Doctor: Mm.
Doctor: 
Doctor: I can imagine, yes. Uh and this has again, been going on for last uh three four days, has it?
Patient: Yeah, yes.
Doctor: OK. Um right, OK. Have you noticed any fevers or temperatures at all?
Patient: Um I I had some some fever in the beginning, but it's now sort of uh going down. I I've been taking some Paracetamol and Ibuprofen for that, and uh they, they help.
Doctor: OK.
Doctor: That's very sensible. Um very good. Um in terms of your chest otherwise, have you any difficulty breathing, or any pain in your chest?
Patient: Um so no pain in chest, but uh I noticed that I um um so I feel a bit winded on exertion. So if I I I haven't been going, going to the gym thus uh, because of that.
Doctor: When you say winded, do you feel, uh do you feel more short of breath would you say, on exertion? Is that when you, OK. Um and and that's mainly on exercise. But when you're resting, there's no problem.
Patient: yeah.
Patient: Yeah, yeah. When when when I'm resting it's OK. So I've been actually going to work, um sort of managing.
Doctor: OK.
Doctor: OK, fine. Just a couple other questions , if you don't mind. Any pain in, in your ears at all?
Patient: Um no.
Doctor: No, you don't feel blocked, or any discharge coming out of your ears?
Patient: 
Patient: No, I don't think so.
Doctor: OK, fine. Um what I'm gonna do, I'm just gonna run through a few uh yes no questions, if you if you don't mind, just a review of your systems. Any, any headaches at all?
Patient: Um, yeah, yeah. That that's still, that's still there.
Doctor: Whereabouts are the head, headaches?
Patient: Um it's sort of general. Uh and uh they're not, they're not always there, but you know every now and then they will come and go.
Doctor: OK. And with those headaches have you had any problems with your, with your eyesight, your vision?
Patient: Um no.
Doctor: Uh any difficulty with seeing bright lights?
Patient: No, I don't think so.
Doctor: No, OK. Um and just moving down now, in terms of any, any feeling nausea or any vomiting?
Patient: No, none of that.
Doctor: OK. Tell me about your bowels. Are your bowels working OK?
Patient: Uh yeah, yeah. Think so.
Doctor: You're passing urine OK?
Patient: Yeah, yeah.
Doctor: How's your appetite, sir?
Patient: Yeah I've been, I've been drinking a lot more than the usual. But uh so other than that, so
Doctor: OK. Um and you're eating and drinking, uh you're eating OK otherwise.
Patient: Um maybe a bit less than uh than what I, I would usually have.
Doctor: OK. Um have you noticed any, any funny rashes at all, on your skin?
Patient: Um no, no.
Doctor: No. And any general muscle pain or aches, joint pain, ?
Patient: Yeah, yeah. Yeah that's, that sort of started in the past couple days.
Doctor: Yeah, OK. Alright um and uh just having a look at the rest of your history, are you otherwise fit and well? Or do you have any other medical problems I should be aware of?
Patient: Um, so otherwise that, uh, I'm I'm fine. There's um, there isn't anything that I'm taking or, I'm being treated for.
Doctor: OK. Um
Patient: Um
Doctor: So you are otherwise fit and well. Uh any, any allergies at all to any medications, I should be aware of?
Patient: Um, no
Doctor: No, OK. Um
Doctor: Any family history at all? So anything relevant in the family that I should be aware of? Anything like diabetes, high blood pressure?
Patient: Um, yeah my grandma has diabetes. Um, and, my grandma.
Doctor: 
Doctor: Your brother?
Doctor: you say grandma, sorry my apologies. Grandma, OK. Uh do you know what type of diabetes it is? Is it type one, type two?
Patient: 
Patient: I think , I think it's type two.
Doctor: OK.
Doctor: Um anything else which you think is significant?
Patient: Yeah, I, I think um, one of my uh, great grandads had an eczema.
Doctor: Excellent, OK. OK. Um just moving on to what we social history, just to get to know you a bit more as a person. Tell me, who do you live with at home?
Patient: Ohh um, I live alone.
Doctor: OK. And you said you're working at the moment?
Patient: Yeah, yeah.
Doctor: What do you do for work?
Patient: I'm an, I'm an accountant.
Doctor: OK, and how's your work going at the moment? Uh over the last, has it, has it been affecting your work?
Patient: Um, a little bit. I've been a bit slower. And you know, it's uh, it's, it's a bit of a stressful period just around the um, end of the year. Right.
Doctor: Mm, OK. Is that stress, is that affecting your um, your mood at all any, in any way?
Patient: Um, no I think I think it's, it's the, you know it's the good kind of stress. It's um, it's good work.
Doctor: OK.
Doctor: Well if there's anything you want to talk to me about, you can always come see me about your mood, um stress, anxiety. happy to help.
Patient: I don't know. I, I like my job. It's just you know, it's a bit more work than usual.
Doctor: OK. OK. And just very briefly, just in terms of smoking uh, do you smoke at all?
Patient: Um, yeah occasionally, you know, cigars and things on company parties.
Doctor: OK so not, not regular. Um and what about alcohol?
Patient: No.
Patient: Um, occasionally yeah I would have some. I'm, I'm not a very big drinker.
Doctor: 
Doctor: socially with work.
Patient: Yeah.
Doctor: Um OK. Um so um
Doctor: just having listened to your story, um uh really just to summarize you know since the last four days you've been feeling generally quite unwell, sore throat, runny nose, bit of a dry cough, bit of muscle pain, weakness.
Doctor: Um had initial fever, but now settled. Um I don't think there's a lot to worry about. I think you probably have, you know a bit of a viral, what we say viral illness, maybe a viral upper respiratory tract infection, or maybe early signs of a flu.
Doctor: Um these normally last about seven to days and just gets better really, um over time. But things you can do to really help yourself, um is get plenty of rest. I'd probably advise you taking a day or two off work if you can. Um
Doctor: Making yourself, pushing fluids and make yourself well-hydrated. Continue with the regular Paracetamol, Ibuprofen. Um and and you should see how things go, really. Um if next week you're still not better, I'd like you to come back and see me.
Doctor: Um is that clear? Does that, does that make, does that make
Patient: Yeah, yeah, that's that's , it makes sense. Uh, I think I'll take a couple days off, and see how it goes.
Doctor: Yeah, yeah.
Doctor: and things to look out for if you're really not getting better, if you if you have a high fever, or your breathing is becoming a bit more labored, or chest pain, I'd like you to come back and see me much sooner, give me a call. Um and we can help you out. OK?
Patient: Yeah.
Patient: Yeah, yeah I understand. Uh, I'll uh, I'll take care.
Doctor: Great. Have a great day. Good luck with your work. Thank you. Bye bye. Bye bye.
Patient: Thank you. Thank you. You too. Bye bye.
    '''
router = APIRouter()

@router.get("/")
def test_with_sample_transcript():
    transcription_doc = process_text(sample_transcript_2)
    print(f"Entities: {transcription_doc.ents}")
    
    keyword_dicts = extract_keywords_descriptors(doc=transcription_doc)
    print(f"Extracted keywords: {keyword_dicts}")
    
    result_dicts = []
    for keyword_dict in keyword_dicts:
        found = False
        print(f"Keyword Dict: {keyword_dict}")
        # Search for an existing entry with the same term.
        for i, existing_dict in enumerate(result_dicts):
            if keyword_dict["term"] == existing_dict["term"]:
                result_dicts[i] = merge_keyword_dicts(existing_dict, keyword_dict)
                print(f"Merged Dicts: {result_dicts}")
                found = True
                break
        if not found:
            result_dicts.append(keyword_dict)
    
    return result_dicts