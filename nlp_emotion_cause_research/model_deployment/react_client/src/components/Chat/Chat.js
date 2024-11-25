import React, { useState, useEffect } from "react";

// Componenents
import TextContainer from '../TextContainer/TextContainer';
import Messages from '../Messages/Messages';
import InfoBar from '../InfoBar/InfoBar';
import Input from '../Input/Input';
import './Chat.css';

// REST API endpoints to ML prediction server
const urlGNE = "http://localhost:8000/api/predict/gne";
const urlEmoCause = "http://localhost:8000/api/predict/emocause";
let introSent = false;

const Chat = () => {

  // ----------------------------------- Initialization -----------------------------------
  useEffect(() => {
    if(!introSent)
      sendIntro()
  });

  // Possible responses to add variation to the model's messaging
  let thinkingMsgs = [
    "Hm let's take a closer look at your message...",
    "Let me think about that...",
    "Give me a second...",
    "Let's see if I can figure this one out...",
    "Hmm I need a moment to think about this one...",
    "Let me check this one out...",
    "Give me a few seconds to figure this out..."
  ]
  let answerMsgs = [
    "Here's what I think about the emotion of your message: ",
    "I give your message a emotion that is: ",
    "After some thinking, I would give your message a emotion that is: ",
    "Based on what I've seen before, I would say your message has this emotion: ",
    "It seems likely to me that your message emotion is: ",
    "The emotion I would give your message is: ",
    "I think the emotion of your message is: "
  ]

  // Init models
  let GNE = {
    name: "GoodNewsEveryone Model", 
    emotion: null, 
    cause: null,
    thinking: "",
    answer: ""
  }
  let EmoCause = {
    name: "EmoCause Model", 
    emotion: null,
    cause: null,
    thinking: "",
    answer: ""
  }

  // Read name from URL on init, parse space characters
  const humanUser = decodeURI(window.location.search.split('=')[1]);                        
  const [users, setUsers] = useState([
    { name: humanUser }, { name: GNE.name }, { name: EmoCause.name }
  ]);

  // Stateful message rendering
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);  

  // Request headers for API call to ML server for predictions
  let params = {
    method: "POST",  
    headers: { "Content-Type": "application/json" },  
    body: null
  }

  // ----------------------------------- Messaging Functions -----------------------------------

  // Generate a random delay between messages the models send
  const getDelay = () => { return Math.floor(Math.random() * (1650 - 2250) ) + 2250;}

  /* Get and return a random response from a list of responses 
      params: 
        - responses, the array of responses to randomly select from
        - notAllowed, the responses that has already been selected (by the other model) or last iteration
  */
  const getResponse = (responses, notAllowed) => {
    let randomResponse = responses[Math.floor(Math.random() * responses.length)];
    while(notAllowed.includes(randomResponse))
      randomResponse = responses[Math.floor(Math.random() * responses.length)];
    return randomResponse;
  }

  // Retrieve GNE emotion prediction from ML server for the user sent text
  const getPredictionGNE = async (userMsg) => {
    params.body = JSON.stringify({ "queries" : [userMsg] });
    await(                                                                                                     
      fetch(urlGNE, params)
      .then((response) => response.json())
      .then((emotionData) => { 
        const { predicted_cause, predicted_emotion } = emotionData.predictions[0]
        GNE.emotion = predicted_emotion
        GNE.cause = predicted_cause
      })
    ) .catch((err) => { console.log(err); }); 
  }
  // Retrieve EmoCause emotion prediction from ML server for the user sent text
  const getPredictionEmoCause = async(userMsg) => {
    params.body = JSON.stringify({ "queries" : [userMsg] });
    await(
      fetch(urlEmoCause, params)
      .then((response) => response.json())
      .then((emotionData) => { 
        const { predicted_cause, predicted_emotion } = emotionData.predictions[0]
        EmoCause.emotion = predicted_emotion
        EmoCause.cause = predicted_cause
      })
    ) .catch((err) => { console.log(err); });
  }

  // Render a message to the screen
  const sendMessage = (name, msg) => {
    let currMsg = { user: name, text: msg};
    setMessages(prevMsgs => ([ ...prevMsgs, currMsg ]));    
  } 

  const respondGNE = async (userMsg, callback) => {
    // Randomly select thinking and answer responses
    GNE.thinking = getResponse(thinkingMsgs, [EmoCause.thinking, GNE.thinking]);
    GNE.answer = getResponse(answerMsgs, [EmoCause.answer, GNE.answer]);

    // Render the models messages
    let thinkDelay = getDelay();
    let repeatDelay = thinkDelay + getDelay();
    let emotionDelay = repeatDelay + getDelay();
    let causeDelay = emotionDelay + getDelay();

    setTimeout(function() { sendMessage(GNE.name, GNE.thinking) }, thinkDelay);                         // Send thinking message

    // Retrieve GNE emotion from the ML prediction server for the message
    await getPredictionGNE(userMsg);

    setTimeout(function() { sendMessage(EmoCause.name, `${EmoCause.answer} "${EmoCause.emotion}"`) }, emotionDelay / 2); // Send the emotion
    setTimeout(function() { sendMessage(EmoCause.name, `This is why I think that: "${EmoCause.cause}"`) }, causeDelay / 2); // Send the cause
    if(callback)
      setTimeout(function() { callback(userMsg, function(){}) }, thinkDelay);                         // Call the other model if not yet called                                 
  }

  const respondEmoCause = async (userMsg, callback) => {
    // Randomly select thinking and answer responses
    EmoCause.thinking = getResponse(thinkingMsgs, [GNE.thinking, EmoCause.thinking]);
    EmoCause.answer = getResponse(answerMsgs, [GNE.answer, EmoCause.answer]);

    // Render the models messages
    let thinkDelay = getDelay();
    let repeatDelay = thinkDelay + getDelay();
    let emotionDelay = repeatDelay + getDelay();
    let causeDelay = emotionDelay + getDelay();
    setTimeout(function() { sendMessage(EmoCause.name, EmoCause.thinking) }, thinkDelay);                       // Send thinking message

    // Retrieve the EmoCause emotion from the ML prediction server for the message
    await getPredictionEmoCause(userMsg);

    setTimeout(function() { sendMessage(EmoCause.name, `${EmoCause.answer} "${EmoCause.emotion}"`) }, emotionDelay / 2); // Send the emotion
    setTimeout(function() { sendMessage(EmoCause.name, `Here's what I saw that makes me think that: "${EmoCause.cause}"`) }, causeDelay / 2); // Send the cause
    if(callback)
      setTimeout(function() { callback(userMsg, function(){}) }, thinkDelay);                         // Call the other model if not yet called                                 
  }

  const handleMessage = async (event) => {
    event.preventDefault();
    // Render user message
    let userMsg = message;
    sendMessage(humanUser, message)
    setMessage('');

    // Randomly alternate which model responds to the message first
    if(Math.random() < 0.5){
      respondGNE(userMsg, respondEmoCause)
    }
    else{
      respondEmoCause(userMsg, respondGNE);
    }
  }

  // Send introduction message once user joins room
  const sendIntro = () => {
    let welcomeDelay = getDelay() / 5;
    let introDelay = welcomeDelay + (getDelay() * 1.5);
    let emotionDelay = introDelay + (getDelay() * 1.5);
    let descriptionDelay = emotionDelay + (getDelay() * 1.5)
    
    setTimeout(function() { 
      sendMessage("Admin", `Welcome to Emotion Cause Extraction (ECE) chat ${humanUser}!`) 
    }, welcomeDelay); 
    setTimeout(function() { 
      sendMessage("Admin", `Send a message and ${GNE.name} and ${EmoCause.name} will try to predict 
      the emotion and extract its cause from your message`) 
    }, introDelay); 
    setTimeout(function() { 
      sendMessage("Admin", `${GNE.name} was trained on 5,000+ unique news headlines and ${EmoCause.name} was trained on 4,613 unique empathic dialogues.`) 
    }, emotionDelay); 
    setTimeout(function() { 
      sendMessage("Admin", `Both models have been trained for ECE and are capable of prediction from a similar set of emotions. Try sending a message!`) 
    }, descriptionDelay); 
    introSent = true;
  }

  // ----------------------------------- Rendered HTML -----------------------------------

  return (
    <div className="outerContainer">
      <div className="container">
          <InfoBar room="ECE Chat.ML"/>
          <Messages messages={messages} name={humanUser} />
          <Input message={message} setMessage={setMessage} sendMessage={handleMessage} />
      </div>
      <TextContainer users={users}/>
    </div>
  );
}

export default Chat;
