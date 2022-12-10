package com.cookandroid.counselingapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.annotations.Nullable;

import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private RecyclerView.Adapter adapter;
    private RecyclerView.LayoutManager layoutManager;
    private List<Chat> chatList;
    private String nickname = "익명2";

    private EditText chatText;
    private Button sendButton;

    private DatabaseReference myRef;


    @SuppressLint("WrongViewCast")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        chatText = findViewById(R.id.chatText);
        sendButton = findViewById(R.id.sendButton);


        sendButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //입력창에 메시지를 입력 후 버튼클릭했을 때
                String msg = chatText.getText().toString();

                if(msg != null){
                    Chat chat = new Chat();
                    chat.setName(nickname);
                    chat.setMsg(msg);

                    //메시지를 파이어베이스에 보냄.
                    // AsyncTask를 통해 HttpURLConnection 수행.
                    String url="http://211.243.154.156:5979/chatbot";
                    NetworkTask networkTask = new NetworkTask(url, msg);
                    networkTask.execute();

                    myRef.push().setValue(chat);

                    recyclerView.scrollToPosition(recyclerView.getAdapter().getItemCount() - 1);

                    chatText.setText("");
                }

            }
        });
        //리사이클러뷰에 어댑터 적용
        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setHasFixedSize(true);
        layoutManager = new LinearLayoutManager(this);
        recyclerView.setLayoutManager(layoutManager);

        chatList = new ArrayList<>();
        adapter = new ChatAdapter(chatList, nickname);
        recyclerView.setAdapter(adapter);

        FirebaseDatabase database = FirebaseDatabase.getInstance();
        myRef = database.getReference("message");

        //데이터들을 추가, 변경, 제거, 이동, 취소
        myRef.addChildEventListener(new ChildEventListener() {
            @Override
            public void onChildAdded(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {

                //어댑터에 DTO추가
                Chat chat = snapshot.getValue(Chat.class);
                ((ChatAdapter)adapter).addChat(chat);
            }

            @Override
            public void onChildChanged(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {

            }

            @Override
            public void onChildRemoved(@NonNull DataSnapshot snapshot) {

            }

            @Override
            public void onChildMoved(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {

            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });
    }

    public class NetworkTask extends AsyncTask<Void, Void, String> {

        private String url;
        private String values;

        public NetworkTask(String url, String values) {

            this.url = url;
            this.values = values;
        }

        @Override
        protected String doInBackground(Void... params) {

            String result; // 요청 결과를 저장할 변수.
            HttpConnection requestHttpURLConnection = new HttpConnection();
            result = requestHttpURLConnection.POSTFunction(url, values); // 해당 URL로 부터 결과물을 얻어온다.
            Log.d("MainActivity","응답"+result);
            return result;
        }

        @Override
        protected void onPostExecute(String s) {
            super.onPostExecute(s);

            Chat chat = new Chat();
            chat.setName("챗봇");
            GetJSON getJSON = new GetJSON();
            String result = getJSON.getJsonData(s);
            chat.setMsg(result);

            myRef.push().setValue(chat);

            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    recyclerView.scrollToPosition(recyclerView.getAdapter().getItemCount() - 1);
                }
            }, 200);


        }
    }





}
