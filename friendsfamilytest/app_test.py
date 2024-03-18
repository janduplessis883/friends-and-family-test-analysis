import time
import numpy as np
import pandas as pd
import streamlit as st

_LOREM_IPSUM = """
Very prompt and careful attention. I am waiting to speak with a GP for an ongoing issue, I'm waiting and can't get any answers. I was also called in but then they couldn't do any tests for an admin issue. At the moment I'm not very happy Yes, getting help to address my issue I'm still getting used to the new premises in two different locations. I find it very difficult to speak to a receptionist by telephone as the waiting time is too long. Quicker response time to incoming calls. I've tried using Patches as suggested to make an appointment and receive a message telling me to call the surgery to make an appointment ! Mostly efficient and quick response. No the doctor I saw didn't seem to have much confidence. I sat down for the blood test and she looked panicked about doing it saying I have very small veins and kept repeating that she doesn't think she'll be able to take blood.. its never been an issue before and hearing the doctor keep saying that and looking flustered isn't exactly comforting. Right after I had a cervical smear and was tense due to the whole interaction and the doctor's confidence so I was having a hard time relaxing. Its not a pleasant experience on any given day and the fact that she kept saying stop tensing in an annoyed tone didn't help. I left the appointment feeling super tense and was not able to do the blood test, I now have to spend another morning going to the hospital to get it done a calmer doctor I saw a very kind very understanding doctor Dr [PERSON] who was extremely helpful and put me at complete ease one thing on a negative note it takes me well over half an hour to talk to this surgery when I ring and today after waiting again over half an hour my call was hung up and I did not get to speak to the receptionist Yes making sure when I wait in line to be answered on the telephone that my call isn't hung up when it is my turn Efficient service Friendly & help where they can You don't see one doctor who knows you for long. You change doctors like clothes. I would have loved to see the same doctor every time l visit because they know your history or problems. My GP, Dr [PERSON], is excellent but I haven’t seen him or any other GP since October 2022. An efficient, friendly surgery. I miss the previous link between patient and particular doctor which certainly in my case proved beneficial, even life-saving. With the present surgery I find the turnover of doctors too great and disconcerting. It's the bad experience I had with the lady at the front desk. My appointment was for 10am and I was there by 10:10 am she turned me away for 10 minutes late. I requested to talk to the nurse since it was my general check up. She said, she sent a email and no reply. She should have taking into account how far I came from and ask gone to ask the nurse if ican be seen but she was in a hurry to turn me away instead. And that was my first time at the 1a address for a check up. I think it was a very poor customer service experience I have ever had. no waiting good rapport Very quick and kind Professional Tea & biscuits The light in the building, sets the tone lightness, professional and care No I was very satisfied Quick and professional The reception person did not acknowledge my presence so I waited until she was ready to communicate. At that point I’d expect a greeting and perhaps recognition I hadn’t interrupted her computer interaction. Not forthcoming. As this was my first visit to this site, I was directed to wait but had to ask where. Fortunately I wasn’t presenting in a stressed mind frame, but this is a communication failure similar to what I remembered from the previous premises in SW10, so not promising. In my consultation there was a primary health care worker who communicated appropriately but also a second female whose presence went completely unexplained and who failed to communicate at all. What was she doing there and who should have introduced her / explained her presence ? Basically, customer communication remains poor at the surgery. I was already thinking of looking elsewhere and that instinct hasn’t changed. 20 years as a patient (previously The Redcliffe Surgery)Both my wife and II have always been very satisfied with the care and attention. Less holding time on the telephone. More opportunity for face-to-face consultation. Closer doctor/patient relationship. clear and friendly No Very happy with the service my surgery provides the GPs are friendly and helpful. I didn't have to wait to long to be see bh the Doctor So far in the time that I'm visiting these centre , I being haven a good service. I don't have to complain for anything . Thank you . Very professional and efficient. A pleasure to interact with. All in all , a absolutely first class. No I don’t use the website. Thank you for your care. I have had a Extremely likely experience with you. Good Overall, it feels like the administration side can be improved significantly, which would make your lives easier and your patients' lives easier. Two examples: (1) It's been a pain trying to schedule vaccinations for my baby daughter - if we try to arrange ahead of time, when prompted by the SMS messages received, then reception staff tell us that the rota isn't yet available and that they'll call us back once it is, but then they don't call back and when we then call to arrange a time, we find that the times are impossible ones (and that sometimes you can spend almost an hour on hold as the second in the queue). (2) I was asked to come in for a blood pressure test, but wasn't seen for an hour despite speaking to reception staff several times, only to finally be told I could use the machine at the back of the reception (which was news to me!). I still then got a call a few hours later telling me I hadn't shown up! Last March from one hour to the other i just could not move at all- which was describerd by doctors through phone it was sciatica - i could not move for like 2 weeks, according to law in the UK GP are not allowed to give steroid injection to help at least to stand up ------ I understood that of course . I was in bed hardly made my daily routine alone , but i managed with 3 boxes of painkillers - then i flew to my home country and then I was walking again within 5 weeks . THen i had the physiotherapy, swimming and yoga and weight bath on my neck for couple of months. I am still not able to go back to the UK or even work normally. BUt the GP did the most He was allowed to help me. I do appreciate still His phone call very much . Thank you very much indeed . THere are laws and rules and gaps , but its not the GP fault. SO im happy with you . Regard,s BEtty [PERSON] Very helpful and supportive discussion - thank you current operating method *lacks* the following: a) Dr - patient continuity of care b) Dr to patient accountability c) Dr - patient personal contact & trust relationship d) Dr - patient initiative & proactive treatment There does not seem to be a particular doctor for a patient any more .I usually was able to see the same doctor when you were at the Redcliffe Surgery . Also have seen random doctors but not from the surgery I find PATCH system really useful, the only thing is that I should have the option to reply without opening another Patch and write everything all over again. Also, to book blood tests, it would be really useful to book a slot online instead of calling the reception, which wastes a lot of time on both sides. Thank you Health Partners is a fantastic GP practice, a real gem. All staff from the receptionist to the nurses to the doctors are extremely responsive and experienced. It’s a great example of how effective a health public service can be run It's Extremely likely the doctors and the personal at the reception very kind and understand me what I need. Now I need an appointment with my doctor. because I have had a stabbing pain in my chest on 2 occasions. I was very scared since my sister discovered that she had a heart attack and they did a triple bypass. and I'm not sure if I make an appointment to see what my doctor thinks or what I should do. I've never use the website Patches improves the situation. I am very happy with all the staff receptionists, nurse and Doctors at the surgery. There is one thing I’m not to happy about is the long wait for a call to the surgery to be answered and twice having waited one and half hours the call was cut off Reception staff are very helpful. Dr [PERSON] is first class. Waiting times are acceptable. The website is horrible. Might it be possible for you to convene a small panel of patients (three or four) to review the website and propose ways to simplify it? [PERSON].....aged 89. Such an impressive efficient team The triage phone back system does not work at all well for people with jobs. Hi there i was never able to see any gp at redcliffe when i was sick so i have not tried to see one at violett melchett. Covid vax experience was great. The only contact i currently have is being periodically chased for my blood pressure! Do you have any special/preventative health checks for >50yo? Thanks I have regular appointments at Chelsea and Westminster who prescribe medicines, blood tests etc for me. I also have medicines prescribed and similar interactions with yourselves. It would be much easier if these were coordinated - for instance blood tests get duplicated, some medicines are available from my local chemists, others can only be obtained at the hospital. It would be much better if these were coordinated. I would also be supportive of proactive patient health care, particularly as one gets older and more prone to illness. Regular health check ups and tests would prevent illness and save costs in the long run. my 
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.002)



    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)


if st.button("Stream data"):
    st.write_stream(stream_data)
