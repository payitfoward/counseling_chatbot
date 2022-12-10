package com.cookandroid.counselingapp;

import android.os.Build;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;


import com.cookandroid.counselingapp.Chat;

import java.util.List;

public class ChatAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private List<Chat> chatList;
    private String name;


    public ChatAdapter(List<Chat> chatData, String name){
        //MainActivity.java에서 받은 데이터들을 저장
        chatList = chatData;
        this.name = name;
    }

    public static class chatBotViewHolder extends RecyclerView.ViewHolder{
        public TextView nameText;
        public TextView msgText;
        public LinearLayout msgLinear;
        public ImageView profile;

        public View rootView;

        public chatBotViewHolder(@NonNull View itemView) {
            super(itemView);

            profile = itemView.findViewById(R.id.profile);
            nameText = itemView.findViewById(R.id.chatBot);
            msgText = itemView.findViewById(R.id.chatBot_message);
            msgLinear = itemView.findViewById(R.id.msgLinear);

            rootView = itemView;

            itemView.setEnabled(true);
            itemView.setClickable(true);
        }
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{
        public TextView user_message;
        public LinearLayout my_msgLinear;

        public View rootView;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);

            user_message= itemView.findViewById(R.id.user_message);
            my_msgLinear=itemView.findViewById(R.id.my_msgLinear);
            rootView = itemView;

            itemView.setEnabled(true);
            itemView.setClickable(true);
        }
    }


    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view;
        switch (viewType){
            case 0:
                view = LayoutInflater.from(parent.getContext()).inflate(R.layout.msg_item,parent,false);
                return new MyViewHolder(view);
            case 1:
                view = LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_message,parent,false);
                return new chatBotViewHolder(view);
        }
        //inflation 과정
        LinearLayout linearLayout = (LinearLayout) LayoutInflater.from(parent.getContext()).inflate(R.layout.chat_message,parent,false);
        MyViewHolder myViewHolder = new MyViewHolder(linearLayout);

        return myViewHolder;
    }


    @Override
    public void onBindViewHolder(@NonNull  RecyclerView.ViewHolder holder, int position) {

        Chat chat = chatList.get(position);


        if(chat.getName().equals(this.name)){
            MyViewHolder myHoler = (MyViewHolder) holder;
            //사용자가 저장된 이름과 같을 시 오른쪽으로 정렬
            myHoler.user_message.setText(chat.getMsg());
            myHoler.user_message.setTextAlignment(View.TEXT_ALIGNMENT_VIEW_START);
            myHoler.user_message.setGravity(Gravity.RIGHT);



            myHoler.my_msgLinear.setGravity(Gravity.RIGHT);
        } else {
            chatBotViewHolder chatHolder = (chatBotViewHolder) holder;
            chatHolder.nameText.setText(chat.getName());
            chatHolder.msgText.setText(chat.getMsg());
            //아닐 시 왼쪽 정렬
            chatHolder.nameText.setTextAlignment(View.TEXT_ALIGNMENT_VIEW_START);
            chatHolder.msgText.setTextAlignment(View.TEXT_ALIGNMENT_VIEW_START);

            chatHolder.msgLinear.setGravity(Gravity.LEFT);
        }


    }

    @Override
    public int getItemViewType(int position) {
        Chat chat = chatList.get(position);

        if (chat.getName().equals(this.name)) {
            return 0;
        } else {
            return 1;
        }
    }

    //메시지아이템 갯수세기
    @Override
    public int getItemCount() {
        return chatList == null ? 0: chatList.size();
    }

    //메시지아이템의 추가 및 적용
    public void addChat(Chat chat){
        chatList.add(chat);
        notifyItemInserted(chatList.size()-1);
    }
}