import React, { useState } from 'react';
import { Link } from "react-router-dom";

import './Join.css';

// Submit if user presses enter key
function checkKey(event){
    if (event.key === 'Enter') {
        document.getElementById("JoinButton").click();
    }
}

export default function SignIn() {
  const [name, setName] = useState('');

  return (
    <div className="joinOuterContainer">
      <div className="joinInnerContainer">
        {/* <img src="logo.png" height="167" width="167"></img> */}
        {/* <img src="https://lh3.googleusercontent.com/d/1C74gG6Y9OZvcClT3c-Ho4X6BG9uRMGTF=s220" height="167" width="167"></img> */}
        <img src="https://lh3.googleusercontent.com/d/1nQsZ_SHHwWj3DgFPBcuIxsCp0pZqM_F0=s220" height="130" width="130"></img>

        <h1 className="heading">ECE Chat.ML</h1>
        <div>
          <input 
            id="NameBox" placeholder="Enter your name" className="joinInput" type="text" 
            onChange={(event) => setName(event.target.value)} onKeyPress={(event) => checkKey(event)}
          />
        </div>
        <Link onClick={e => (!name) ? e.preventDefault() : null} to={`/chat?name=${name}`}>
          <button id="JoinButton" className={'button mt-20'} type="submit">Start Chatting</button>
        </Link>
      </div>
    </div>
  );
}
